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
/// Maintains up-to-16 per-tenant FIFOs internally (4-bit tenantId after the
/// design v0.2.1 bit-layout change in fifo_device.hpp; bit 63 is reserved for
/// the FIFO push() XOR and must not be touched by business fields). On every
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
  TenantAwareProxyHandler(ContextProxyHandler inner, PolicyMode mode = PolicyMode::SinglePassthrough)
      : inner_(std::move(inner)), mode_(mode) {}

  /// Re-point the inner handler (used by TenantAwareProxyService, where the
  /// handler is created before the inner is known).
  void setInner(ContextProxyHandler inner) { inner_ = std::move(inner); }

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

  /// Returns a callable usable as a ContextProxyHandler (std::function compatible).
  ContextProxyHandler asHandler() {
    return [this](ProxyTrigger t, ProxyFifoContext ctx) -> ProxyHandlerResult {
      return this->operator()(t, ctx);
    };
  }

  /// Direct invocation (design.md §5.5 / §5.7, v0.2.1).
  ///
  /// Dispatch order is critical — see iter5 header comment:
  ///   1. parse tenantId
  ///   2. fail-open unregistered tenants to DEFAULT_TENANT
  ///   3. update activeMask (so this very trigger's tenant counts toward bypass)
  ///   4. bypass iff active<=1 AND every per-tenant queue is empty
  ///   5. otherwise enqueue (carrying ORIGINAL fifoPos from ctx) + drain
  ProxyHandlerResult operator()(ProxyTrigger trig, ProxyFifoContext ctx) {
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
    //     before policyd has finished tenant registration. We also bump a
    //     counter so callers (benchmarks, policyd) can detect this race.
    uint32_t regMask = registeredMask_.load(std::memory_order_acquire);
    if (regMask != 0 && (regMask & (uint32_t{1} << tid)) == 0) {
      unregisteredCount_.fetch_add(1, std::memory_order_relaxed);
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
        // Bypass: pass the ORIGINAL ctx (poll-time fifoPos) straight through.
        return inner_(trig, ctx);
      }
    }

    // (5) Slow path: enqueue with monotonic sequence + connection key, drain.
    //     CRITICAL: the PendingTrigger stores the *push-time* fifoPos from
    //     ctx, NOT a fresh fifo->tail() — that's the whole point of the
    //     ContextProxyHandler ABI (design.md §5.7).
    {
      std::lock_guard<std::mutex> g(mu_);
      uint64_t seq = nextSeq_++;
      uint32_t connKey = static_cast<uint32_t>(trig.fields.semaphoreId);
      queues_[tid].push_back(PendingTrigger{trig, seq, connKey, ctx});
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

  /// Test-only: enqueue a trigger WITHOUT draining. Lets a unit test
  /// stockpile triggers so the DRR / aging / per-conn ordering paths can be
  /// exercised against a non-trivial queue state — in production the proxy
  /// thread always calls operator() which drains immediately after enqueue,
  /// so synchronous unit tests can't naturally see deep queues.
  void enqueueForTest(ProxyTrigger trig, ProxyFifoContext ctx) {
    uint32_t tid = static_cast<uint32_t>(trig.fields.tenantId);
    if (tid >= MAX_TENANTS) tid = static_cast<uint32_t>(DEFAULT_TENANT);
    activeMask_.fetch_or(uint32_t{1} << tid, std::memory_order_acq_rel);
    std::lock_guard<std::mutex> g(mu_);
    uint64_t seq = nextSeq_++;
    uint32_t connKey = static_cast<uint32_t>(trig.fields.semaphoreId);
    queues_[tid].push_back(PendingTrigger{trig, seq, connKey, ctx});
    connQueues_[connKey].push_back(seq);
  }

 private:
  struct PendingTrigger {
    ProxyTrigger     trigger;
    uint64_t         seq;       // monotonic enqueue order (proxy.cc polls FIFO in order)
    uint32_t         connKey;   // semaphoreId — same PortChannel shares one proxy sem
    ProxyFifoContext ctx;       // PUSH-time fifoPos + enqueueNs (design.md §5.7)
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
    // Outside the queue lock: dispatch with the ORIGINAL push-time context.
    // This is the v0.2.1 invariant: TriggerSync flush boundaries follow the
    // push-time fifoPos, not the dispatch-time tail (which has advanced by
    // however many triggers were polled while this one waited in queue).
    return inner_(picked.trigger, picked.ctx);
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

  // Caller must hold mu_. Returns the trigger size encoded in fst's low bits.
  static uint64_t headSize(const PendingTrigger& p) {
    return static_cast<uint64_t>(p.trigger.fst & ((uint64_t{1} << TriggerBitsSize) - 1));
  }

  // Caller must hold mu_. DRR quantum per tenant in bytes — see design.md §5.3.1.
  // Quantum × weight is the per-round credit; small enough that long messages
  // span multiple rounds (which is how weight ratios kick in for hetero loads),
  // large enough that pure-small-message workloads don't churn.
  static constexpr uint64_t kDrrQuantumBytes = 64 * 1024;  // 64 KiB

  // Aging threshold for StrictPriority — design.md §5.4.1 specifies 100 ms.
  // After this many ns, a queued trigger's effective priority is boosted by
  // one level per threshold elapsed, capped at the top QoS class. Aging is
  // ONLY about preventing starvation; it never demotes anything.
  static constexpr uint64_t kAgingThresholdNs = 100ULL * 1000ULL * 1000ULL;  // 100 ms

  static uint64_t monoNowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }

  // Caller must hold mu_. Apply the configured PolicyMode but only consider
  // tenants whose queue head is currently eligible per-connection.
  //
  // Side-effects on Fair / Hybrid path: may credit `deficit_[t] += quantum`
  // for one or more tenants in order to find a candidate. This is correct DRR
  // semantics — a tenant that didn't get credited last round will receive its
  // share once a candidate is found.
  uint32_t pickNextLocked() {
    switch (mode_) {
      case PolicyMode::SinglePassthrough:
        for (uint32_t t = 0; t < MAX_TENANTS; ++t)
          if (tenantHeadEligibleLocked(t)) return t;
        return MAX_TENANTS;

      case PolicyMode::StrictPriority:
        return pickStrictPriorityLocked();

      case PolicyMode::Hybrid: {
        // Premium / Realtime first (with aging), otherwise fall through to DRR.
        uint32_t premium = pickStrictPriorityLocked(/*minClass=*/QoSClass::Premium);
        if (premium < MAX_TENANTS) return premium;
        return pickDrrLocked();
      }

      case PolicyMode::Fair:
        return pickDrrLocked();
    }
    return MAX_TENANTS;
  }

  // StrictPriority with aging.
  // - minClass (default BestEffort) lets Hybrid restrict the search to
  //   Premium/Realtime tenants.
  // - Aging promotes a tenant's effective QoS class by 1 per kAgingThresholdNs
  //   the head trigger has been queued, capped at Realtime.
  // - Eligible only when the head is at the front of its connection queue
  //   (design.md §5.7.1).
  uint32_t pickStrictPriorityLocked(QoSClass minClass = QoSClass::BestEffort) {
    uint32_t best = MAX_TENANTS;
    int bestEffPri = -1;
    uint64_t now = monoNowNs();
    for (uint32_t t = 0; t < MAX_TENANTS; ++t) {
      if (!tenantHeadEligibleLocked(t)) continue;
      int basePri = static_cast<int>(tenants_[t].qos_class);
      if (basePri < static_cast<int>(minClass)) continue;

      // Aging boost based on the OLDEST queued trigger for this tenant
      // (= queue head, since we enqueue in arrival order).
      uint64_t enqueueNs = queues_[t].front().ctx.enqueueNs;
      int agedBoost = 0;
      if (enqueueNs != 0 && now > enqueueNs) {
        uint64_t waitNs = now - enqueueNs;
        if (waitNs >= kAgingThresholdNs) {
          agedBoost = static_cast<int>(waitNs / kAgingThresholdNs);
        }
      }
      // Cap effective priority at the top class.
      constexpr int kMaxQosLevel = static_cast<int>(QoSClass::Realtime);
      int effPri = std::min(basePri + agedBoost, kMaxQosLevel);

      if (effPri > bestEffPri) {
        bestEffPri = effPri;
        best = t;
      }
    }
    return best;
  }

  // Real DRR (Deficit Round Robin) — design.md §5.3.1.
  //
  // Classic DRR is STICKY per tenant: once a tenant's turn starts, the
  // scheduler keeps dispatching from that tenant as long as its deficit
  // covers the head message. Only when the head no longer fits does the
  // turn end and we advance to the next eligible tenant (who is then
  // credited `weight × quantum` fresh).
  //
  // This is what makes weight ratios actually translate to bandwidth ratios
  // for identical-size messages: tenant with weight=3 drains ~3× as many
  // messages per turn as a tenant with weight=1.
  //
  // State carried across calls: `drrCurrent_` (-1 = no active turn) and
  // `deficit_[]`. Each pickDrrLocked returns the current tenant if it can
  // still afford its head; otherwise advances.
  uint32_t pickDrrLocked() {
    // Helper: does tenant t currently have a head whose size fits in deficit?
    auto canServe = [this](uint32_t t) -> bool {
      if (!tenantHeadEligibleLocked(t)) return false;
      return deficit_[t] >= static_cast<int64_t>(headSize(queues_[t].front()));
    };

    // Phase 1: if there's an active turn (drrCurrent_) and that tenant can
    // still afford its head, stay on them.
    if (drrCurrent_ < MAX_TENANTS && canServe(drrCurrent_)) {
      uint32_t t = drrCurrent_;
      deficit_[t] -= static_cast<int64_t>(headSize(queues_[t].front()));
      return t;
    }

    // Phase 2: current tenant's turn is over (queue empty, or deficit < head
    // size). Advance to the next eligible tenant. We may need to "skip" non-
    // eligible tenants and credit candidates multiple rounds if message sizes
    // outsize quantum × weight. Bounded by kMaxCreditRounds to handle big
    // outlier messages.
    static constexpr int kMaxCreditRounds = 64;
    uint32_t startCursor = (drrCurrent_ < MAX_TENANTS) ? ((drrCurrent_ + 1) % MAX_TENANTS) : drrCursor_;

    for (int round = 0; round < kMaxCreditRounds; ++round) {
      // Walk tenants in round-robin from startCursor.
      for (uint32_t step = 0; step < MAX_TENANTS; ++step) {
        uint32_t t = (startCursor + step) % MAX_TENANTS;
        if (!tenantHeadEligibleLocked(t)) continue;
        // Start a new turn for t: credit them weight × quantum.
        // (Credit happens exactly once per turn; the deficit carries over
        //  any leftover from previous turns so long-term fairness holds.)
        uint64_t w = tenants_[t].weight ? tenants_[t].weight : 1;
        deficit_[t] += static_cast<int64_t>(w * kDrrQuantumBytes);
        if (canServe(t)) {
          drrCurrent_ = t;
          drrCursor_ = (t + 1) % MAX_TENANTS;
          deficit_[t] -= static_cast<int64_t>(headSize(queues_[t].front()));
          return t;
        }
        // Credit didn't cover — tenant t's head is bigger than weight×quantum.
        // Keep iterating; the next round will credit them again, but try
        // others first (in case someone smaller can be served).
      }
      // Nothing served this scan. If at least one tenant had eligible work
      // (handled by the credit loop above), keep iterating so deficits grow
      // big enough to cover their head.
    }

    // Escape hatch: outlier message larger than 64 × max(weight) × quantum.
    // Pick any eligible tenant unconditionally; don't let deficit grow
    // without bound — clamp to 0 here.
    for (uint32_t step = 0; step < MAX_TENANTS; ++step) {
      uint32_t t = (startCursor + step) % MAX_TENANTS;
      if (tenantHeadEligibleLocked(t)) {
        drrCurrent_ = t;
        drrCursor_ = (t + 1) % MAX_TENANTS;
        deficit_[t] = 0;
        return t;
      }
    }
    drrCurrent_ = MAX_TENANTS;
    return MAX_TENANTS;
  }

 public:
  /// MT-MSCCL++ (design.md §5.3.3 v0.2.1): progress hook called from the
  /// proxy thread's progressHandler. Lets the scheduler retry dispatch
  /// even when no new trigger has arrived — important when a tenant's
  /// token bucket was dry and has since refilled, or when aging must
  /// trigger a re-pick under sustained Realtime traffic.
  /// Safe to call concurrently with operator() since both are invoked on
  /// the single proxy thread.
  void tickProgress() {
    // Cheap fast path: no queued work.
    {
      std::lock_guard<std::mutex> g(mu_);
      if (allQueuesEmptyLocked()) return;
    }
    // Attempt one drain. drain() locks internally and dispatches at most
    // one trigger; the proxy thread will keep calling tickProgress() each
    // poll loop iteration.
    drain();
  }

  /// Counter of how many triggers were observed with a tenantId that
  /// wasn't in registeredMask_. Each such trigger is rewritten to
  /// DEFAULT_TENANT (fail-open, design.md §5.5). Used by benchmarks /
  /// policyd to detect that GPU kernels ran ahead of tenant registration.
  uint64_t unregisteredTriggerCount() const {
    return unregisteredCount_.load(std::memory_order_relaxed);
  }

 private:
  ContextProxyHandler inner_;
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
  // DRR deficit (signed; can be temporarily negative after the escape hatch).
  std::array<int64_t, MAX_TENANTS> deficit_{};
  uint64_t nextSeq_ = 0;
  uint32_t rrCursor_ = 0;
  uint32_t drrCursor_ = 0;
  // DRR active turn: which tenant currently "owns" the dispatch cursor.
  // MAX_TENANTS sentinel = no active turn (idle).
  uint32_t drrCurrent_ = MAX_TENANTS;
  std::atomic<uint64_t> unregisteredCount_{0};
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
        handler_(std::make_shared<TenantAwareProxyHandler>(ContextProxyHandler{}, mode)) {
    // The decorator captures handler_ (constructed BEFORE this lambda runs)
    // and installs it as the proxy's outer context-aware handler.
    // Using setContextHandlerDecorator (not setHandlerDecorator) is what
    // wires the push-time fifoPos through to the inner handleTrigger —
    // see design.md §5.7 (v0.2.1) for why this matters.
    auto h = handler_;
    setContextHandlerDecorator([h](ContextProxyHandler inner) {
      h->setInner(std::move(inner));
      return h->asHandler();
    });
    // MT-MSCCL++ (design.md §5.3.3 v0.2.1): chain a tenant progress tick
    // AFTER the base ProxyService::progressFlushes(). The proxy thread
    // calls this every iteration of the poll loop, so a rate-limited
    // tenant whose token bucket has just refilled (but no new trigger has
    // arrived) still gets drained promptly instead of waiting for the
    // next FIFO push. Same for aging-promoted BestEffort triggers — the
    // tick re-evaluates priorities every loop.
    setExtraProgressHook([h]() { h->tickProgress(); });
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

  /// MT-MSCCL++: total count of triggers that arrived with an unregistered
  /// tenantId during this service's lifetime. Each was rewritten to
  /// DEFAULT_TENANT (fail-open). > 0 indicates kernels ran ahead of policyd
  /// registration; sustained growth is a deeper config bug.
  uint64_t unregisteredTriggerCount() const { return handler_->unregisteredTriggerCount(); }

 private:
  std::shared_ptr<TenantAwareProxyHandler> handler_;
};

}  // namespace tenant
}  // namespace ext
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_TENANT_AWARE_PROXY_HPP_
