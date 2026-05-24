// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// MT-MSCCL++ TenantAwareProxyHandler — header-only reference implementation.
//
// Decorates an mscclpp::ProxyHandler so the host-side proxy multiplexes by
// tenant_id (extracted from ProxyTrigger.fields.tenantId, set by GPU-side
// channel handles in include/mscclpp/port_channel_device.hpp).
//
// Design alignment with doc/design.md:
//   §3.5 — TenantContext / PolicyTable / BandwidthBudget data model
//   §5.3 — Weighted Fair Queuing (Deficit Round Robin) over a per-tenant queue
//   §5.4 — Strict Priority + Aging (chunk-boundary preemption)
//   §5.5 — Single-tenant bypass (zero overhead when popcount(active_mask)==1)
//   §6.3 — Token-bucket rate limiter
//
// Iter 2 ships this surface; the Python prototype in python/mscclpp/ext/tenant.py
// implements the same semantics today (used by the benchmark). Iter 3 wired the
// proxy thread through this class; iter 4 added per-tenant channel injection.
//
// Iter 5 (this revision) hardens the host-side data plane per design v0.2.1:
//   - dispatch order fix: parse tenantId / fail-open / update activeMask BEFORE
//     the bypass decision. Otherwise a brand-new tenant's first trigger is
//     bypassed before activeMask can register it, breaking isolation forever.
//   - per-connection program order invariant: scheduler may reorder across
//     tenants but MUST preserve FIFO order within a single connection
//     (keyed by semaphoreId — same PortChannel = same proxy sem).
//   - TriggerSync cannot pass older same-connection Data/Flag triggers, so
//     flush boundaries stay correct under delayed dispatch.

#ifndef MSCCLPP_EXT_TENANT_AWARE_PROXY_HPP_
#define MSCCLPP_EXT_TENANT_AWARE_PROXY_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <unordered_map>

#include <mscclpp/ext/tenant.hpp>
#include <mscclpp/fifo_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/proxy.hpp>

namespace mscclpp {
namespace ext {
namespace tenant {

/// Token-bucket rate limiter (design.md §6.3.1).
class TokenBucket {
 public:
  TokenBucket() = default;
  TokenBucket(uint64_t refillBps, uint64_t burstBytes)
      : refillBps_(refillBps), burstBytes_(burstBytes), tokens_(static_cast<double>(burstBytes)) {
    lastNs_ = nowNs();
  }

  // Non-copyable, non-movable (std::mutex member).
  TokenBucket(const TokenBucket&) = delete;
  TokenBucket& operator=(const TokenBucket&) = delete;

  /// Reset parameters in place (avoids copy-assign that std::mutex disallows).
  void reset(uint64_t refillBps, uint64_t burstBytes) {
    std::lock_guard<std::mutex> g(mu_);
    refillBps_ = refillBps;
    burstBytes_ = burstBytes;
    tokens_ = static_cast<double>(burstBytes);
    lastNs_ = nowNs();
  }

  /// Try to consume `n` bytes. Returns true if allowed, false if bucket is dry.
  bool tryConsume(uint64_t n) {
    std::lock_guard<std::mutex> g(mu_);
    refillLocked();
    if (refillBps_ == 0 || tokens_ >= static_cast<double>(n)) {
      tokens_ -= static_cast<double>(n);
      if (tokens_ < 0) tokens_ = 0;
      return true;
    }
    return false;
  }

 private:
  static uint64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  void refillLocked() {
    if (refillBps_ == 0) {
      tokens_ = static_cast<double>(burstBytes_);
      return;
    }
    uint64_t now = nowNs();
    double dt = static_cast<double>(now - lastNs_) / 1e9;
    if (dt <= 0) return;
    tokens_ = std::min(static_cast<double>(burstBytes_), tokens_ + dt * static_cast<double>(refillBps_));
    lastNs_ = now;
  }

  uint64_t refillBps_ = 0;
  uint64_t burstBytes_ = 0;
  double tokens_ = 0;
  uint64_t lastNs_ = 0;
  std::mutex mu_;
};

/// Tenant-aware decorator over a `ProxyHandler`.
///
/// Maintains up-to-32 per-tenant FIFOs internally (5-bit tenantId in the
/// current code; design v0.2 narrows this to 4 bits / MAX_TENANTS=16). On every
/// trigger it:
///   (1) extracts trigger.fields.tenantId,
///   (2) fail-open: an unregistered tenantId is rewritten to DEFAULT_TENANT
///       so a kernel that beats policyd registration cannot deadlock,
///   (3) records the tenant in activeMask_ BEFORE checking bypass,
///   (4) bypasses scheduling iff popcount(activeMask)<=1 AND all queues empty,
///   (5) otherwise enqueues per (tenantId, connectionKey) and drains via the
///       configured PolicyMode while preserving same-connection program order.
///
/// Returns Continue/Stop matching the inner handler's last call.
class TenantAwareProxyHandler {
 public:
  TenantAwareProxyHandler(ProxyHandler inner, PolicyMode mode = PolicyMode::SinglePassthrough)
      : inner_(std::move(inner)), mode_(mode) {}

  /// Re-point the inner handler (used by TenantAwareProxyService, where the
  /// handler is created before the inner is known).
  void setInner(ProxyHandler inner) { inner_ = std::move(inner); }

  /// Register / update a tenant. Thread-safe.
  void updateTenant(const TenantContext& ctx, const BandwidthBudget& budget) {
    if (ctx.tenant_id >= MAX_TENANTS) return;
    std::lock_guard<std::mutex> g(mu_);
    tenants_[ctx.tenant_id] = ctx;
    budgets_[ctx.tenant_id] = budget;
    buckets_[ctx.tenant_id].reset(budget.bytes_per_second, budget.burst_bytes);
    registeredMask_.fetch_or(uint32_t{1} << ctx.tenant_id, std::memory_order_release);
  }

  void removeTenant(TenantId id) {
    if (id >= MAX_TENANTS) return;
    registeredMask_.fetch_and(~(uint32_t{1} << id), std::memory_order_release);
    activeMask_.fetch_and(~(uint32_t{1} << id), std::memory_order_release);
  }

  void setMode(PolicyMode mode) { mode_ = mode; }
  PolicyMode mode() const { return mode_; }

  /// Returns a callable usable as a ProxyHandler (std::function compatible).
  ProxyHandler asHandler() {
    return [this](ProxyTrigger t) -> ProxyHandlerResult { return this->operator()(t); };
  }

  /// Direct invocation (design.md §5.5).
  ///
  /// Dispatch order is critical — see iter5 header comment:
  ///   1. parse tenantId
  ///   2. fail-open unregistered tenants to DEFAULT_TENANT
  ///   3. update activeMask (so this very trigger's tenant counts toward bypass)
  ///   4. bypass iff active<=1 AND every per-tenant queue is empty
  ///   5. otherwise enqueue + drain
  ProxyHandlerResult operator()(ProxyTrigger trig) {
    if (!inner_) {
      // Misconfigured: setHandlerDecorator must wire inner_ before triggers flow.
      fprintf(stderr, "[mt-mscclpp] FATAL: TenantAwareProxyHandler inner_ is empty\n");
      return ProxyHandlerResult::Stop;
    }

    // (1) Parse tenant from the trigger's business payload (already restored
    //     from the FIFO MSB flip by ProxyService poll, see design.md §5.2).
    uint32_t tid = static_cast<uint32_t>(trig.fields.tenantId);
    if (tid >= MAX_TENANTS) tid = static_cast<uint32_t>(DEFAULT_TENANT);

    // (2) Fail-open: unregistered tenants are reattributed to DEFAULT_TENANT.
    //     Hard-aborting here would deadlock workloads whose kernels start
    //     before policyd has finished tenant registration.
    uint32_t regMask = registeredMask_.load(std::memory_order_acquire);
    if (regMask != 0 && (regMask & (uint32_t{1} << tid)) == 0) {
      // Rate-limit warning to avoid log floods.
      auto now = std::chrono::steady_clock::now();
      if (now - lastUnregisteredWarnAt_ >= std::chrono::seconds(1)) {
        lastUnregisteredWarnAt_ = now;
        fprintf(stderr, "[mt-mscclpp] WARN: trigger from unregistered tenant %u, "
                        "rewriting to DEFAULT_TENANT (further warnings suppressed for 1s)\n",
                tid);
      }
      tid = static_cast<uint32_t>(DEFAULT_TENANT);
    }

    // (3) Record this tenant as active BEFORE checking bypass. The fetch_or
    //     return value is the OLD mask; we OR in the bit locally so step (4)
    //     sees a value that includes this very tenant.
    uint32_t oldMask = activeMask_.fetch_or(uint32_t{1} << tid, std::memory_order_acq_rel);
    uint32_t mask = oldMask | (uint32_t{1} << tid);

    // (4) Single-tenant bypass: zero scheduling overhead, BUT only when the
    //     scheduler has no in-flight work. If we bypassed while queues still
    //     held older triggers we would reorder relative to FIFO push order.
    if (__builtin_popcount(mask) <= 1) {
      bool queuesEmpty;
      {
        std::lock_guard<std::mutex> g(mu_);
        queuesEmpty = allQueuesEmptyLocked();
      }
      if (queuesEmpty) {
        return inner_(trig);
      }
    }

    // (5) Slow path: enqueue with monotonic sequence + connection key, drain.
    {
      std::lock_guard<std::mutex> g(mu_);
      uint64_t seq = nextSeq_++;
      uint32_t connKey = static_cast<uint32_t>(trig.fields.semaphoreId);
      queues_[tid].push_back(PendingTrigger{trig, seq, connKey});
      // Per-connection arrival order: this seq is now the youngest pending
      // trigger on connKey. The matching pop happens in drain() once this
      // trigger is actually dispatched.
      connQueues_[connKey].push_back(seq);
    }
    return drain();
  }

  /// Test-only accessor: total pending PendingTrigger across all tenant queues.
  /// Used by unit tests for the per-connection ordering invariant.
  size_t pendingCountForTest() {
    std::lock_guard<std::mutex> g(mu_);
    size_t n = 0;
    for (auto& q : queues_) n += q.size();
    return n;
  }

 private:
  struct PendingTrigger {
    ProxyTrigger trigger;
    uint64_t     seq;          // monotonic enqueue order (proxy.cc polls FIFO in order)
    uint32_t     connKey;      // semaphoreId — same PortChannel shares one proxy sem
  };

  ProxyHandlerResult drain() {
    // The proxy thread is single-threaded; operator() is called once per
    // trigger and immediately drains. We process AT MOST one trigger here
    // and release the queue lock before invoking inner_, which keeps the
    // mutex hold time bounded (inner_ may block on conn.flush()).
    PendingTrigger picked{};
    uint32_t pickedTid = MAX_TENANTS;
    uint64_t size = 0;
    {
      std::lock_guard<std::mutex> g(mu_);
      pickedTid = pickNextLocked();
      if (pickedTid >= MAX_TENANTS) return ProxyHandlerResult::Continue;
      auto& q = queues_[pickedTid];
      if (q.empty()) return ProxyHandlerResult::Continue;
      picked = q.front();
      size = static_cast<uint64_t>(picked.trigger.fst & ((uint64_t{1} << TriggerBitsSize) - 1));
      // Rate limit check (bucket has its own internal mutex; cheap).
      if (!buckets_[pickedTid].tryConsume(size == 0 ? 1 : size)) {
        // Bucket dry — keep trigger queued; we'll retry on next operator() call.
        return ProxyHandlerResult::Continue;
      }
      q.pop_front();
      // Per-connection ordering: pickNextLocked() only returned a trigger
      // whose seq is the current head of connQueues_[connKey], so popping
      // the front here keeps the invariant.
      auto& cq = connQueues_[picked.connKey];
      // Defensive: front should equal picked.seq.
      if (!cq.empty() && cq.front() == picked.seq) {
        cq.pop_front();
      }
    }
    // Outside the queue lock: process the trigger.
    return inner_(picked.trigger);
  }

  // Caller must hold mu_. True iff every tenant queue is empty.
  bool allQueuesEmptyLocked() const {
    for (const auto& q : queues_)
      if (!q.empty()) return false;
    return true;
  }

  // Caller must hold mu_. Returns true iff the front of queues_[tid] is
  // eligible to dispatch right now under the per-connection ordering
  // invariant — i.e., its seq is the head of its connKey's connQueue.
  bool tenantHeadEligibleLocked(uint32_t tid) const {
    const auto& q = queues_[tid];
    if (q.empty()) return false;
    const auto& head = q.front();
    auto it = connQueues_.find(head.connKey);
    if (it == connQueues_.end() || it->second.empty()) return true;  // defensive
    return it->second.front() == head.seq;
  }

  // Caller must hold mu_. Apply the configured PolicyMode but only consider
  // tenants whose queue head is currently eligible per-connection.
  uint32_t pickNextLocked() {
    switch (mode_) {
      case PolicyMode::SinglePassthrough:
        for (uint32_t t = 0; t < MAX_TENANTS; ++t)
          if (tenantHeadEligibleLocked(t)) return t;
        return MAX_TENANTS;

      case PolicyMode::StrictPriority: {
        uint32_t best = MAX_TENANTS;
        int bestPri = -1;
        for (uint32_t t = 0; t < MAX_TENANTS; ++t) {
          if (!tenantHeadEligibleLocked(t)) continue;
          int pri = static_cast<int>(tenants_[t].qos_class);
          if (pri > bestPri) {
            bestPri = pri;
            best = t;
          }
        }
        return best;
      }

      case PolicyMode::Hybrid: {
        // Premium / Realtime first; otherwise DRR.
        for (uint32_t t = 0; t < MAX_TENANTS; ++t) {
          if (!tenantHeadEligibleLocked(t)) continue;
          if (tenants_[t].qos_class >= QoSClass::Premium) return t;
        }
        // fall through
        [[fallthrough]];
      }
      case PolicyMode::Fair: {
        // Simple round-robin among eligible queues; weight applied via
        // deficit counter would go here in a richer impl.
        for (uint32_t step = 0; step < MAX_TENANTS; ++step) {
          uint32_t t = (rrCursor_ + step) % MAX_TENANTS;
          if (tenantHeadEligibleLocked(t)) {
            rrCursor_ = (t + 1) % MAX_TENANTS;
            return t;
          }
        }
        return MAX_TENANTS;
      }
    }
    return MAX_TENANTS;
  }

  ProxyHandler inner_;
  std::atomic<uint32_t> activeMask_{0};
  std::atomic<uint32_t> registeredMask_{0};
  PolicyMode mode_;
  std::mutex mu_;
  std::array<std::deque<PendingTrigger>, MAX_TENANTS> queues_{};
  // Per-connection FIFO of pending seq numbers — order of arrival on each
  // semaphoreId. Heterogeneous map so we only pay for actually-used keys.
  std::unordered_map<uint32_t, std::deque<uint64_t>> connQueues_;
  std::array<TenantContext, MAX_TENANTS> tenants_{};
  std::array<BandwidthBudget, MAX_TENANTS> budgets_{};
  std::array<TokenBucket, MAX_TENANTS> buckets_{};
  uint64_t nextSeq_ = 0;
  uint32_t rrCursor_ = 0;
  std::chrono::steady_clock::time_point lastUnregisteredWarnAt_{};
};

/// A ProxyService that runs TenantAwareProxyHandler in the proxy thread.
///
/// Use exactly like ProxyService — `addSemaphore`, `addMemory`, `portChannel`,
/// `startProxy`, `stopProxy` are inherited. The MT extension is:
///
///   svc.updateTenant({tid, qos, weight, ...}, {refill_bps, burst});
///   svc.setMode(PolicyMode::Fair);
///
/// Once `startProxy()` is called the proxy thread automatically multiplexes
/// incoming triggers by `trigger.fields.tenantId` (set by the GPU-side channel
/// handles via include/mscclpp/port_channel_device.hpp).
///
/// CRITICAL for CUDA-graph capture: because all scheduling happens INSIDE the
/// proxy thread, the GPU side and host launch site see no extra synchronization
/// or callbacks per kernel — so the multi-tenant path is fully graph-capturable
/// in iter 3 (fixes iter 2 limitation #1).
class TenantAwareProxyService : public ProxyService {
 public:
  TenantAwareProxyService(PolicyMode mode = PolicyMode::SinglePassthrough, int fifoSize = DEFAULT_FIFO_SIZE)
      : ProxyService(fifoSize),
        handler_(std::make_shared<TenantAwareProxyHandler>(ProxyHandler{}, mode)) {
    // The decorator captures handler_ (constructed BEFORE this lambda runs)
    // and installs it as the proxy's outer handler.
    auto h = handler_;
    setHandlerDecorator([h](ProxyHandler inner) {
      h->setInner(std::move(inner));
      return h->asHandler();
    });
  }

  /// Register or update a tenant's policy and rate-limit budget. Thread-safe
  /// and may be called while the proxy is running.
  void updateTenant(const TenantContext& ctx, const BandwidthBudget& budget) { handler_->updateTenant(ctx, budget); }

  /// Convenience: register a tenant with rate-limit specified directly.
  void registerTenant(TenantId tenantId, QoSClass qos, uint32_t weight, uint64_t bandwidthMaxBps = 0,
                      uint64_t burstBytes = 0) {
    TenantContext ctx{tenantId, qos, weight, 0, 0, bandwidthMaxBps, 0};
    BandwidthBudget bud{bandwidthMaxBps, burstBytes ? burstBytes : (bandwidthMaxBps ? bandwidthMaxBps / 10 : 0), 0};
    handler_->updateTenant(ctx, bud);
  }

  void removeTenant(TenantId tenantId) { handler_->removeTenant(tenantId); }

  void setMode(PolicyMode mode) { handler_->setMode(mode); }
  PolicyMode mode() const { return handler_->mode(); }

 private:
  std::shared_ptr<TenantAwareProxyHandler> handler_;
};

}  // namespace tenant
}  // namespace ext
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_TENANT_AWARE_PROXY_HPP_
