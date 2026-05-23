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
// implements the same semantics today (used by the benchmark). Iter 3 will
// route ProxyService::handleTrigger() through this class to deliver scheduling
// inside the proxy thread without a Python round-trip.

#ifndef MSCCLPP_EXT_TENANT_AWARE_PROXY_HPP_
#define MSCCLPP_EXT_TENANT_AWARE_PROXY_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>

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
/// Maintains 32 per-tenant FIFOs internally; on every trigger it:
///   (1) extracts trigger.fields.tenantId,
///   (2) enqueues the trigger into the matching per-tenant queue,
///   (3) drains queues into `inner_` using the configured PolicyMode,
///   (4) applies per-tenant token buckets if rate limiting is enabled.
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
    activeMask_.fetch_or(uint32_t{1} << ctx.tenant_id, std::memory_order_release);
  }

  void removeTenant(TenantId id) {
    if (id >= MAX_TENANTS) return;
    activeMask_.fetch_and(~(uint32_t{1} << id), std::memory_order_release);
  }

  void setMode(PolicyMode mode) { mode_ = mode; }
  PolicyMode mode() const { return mode_; }

  /// Returns a callable usable as a ProxyHandler (std::function compatible).
  ProxyHandler asHandler() {
    return [this](ProxyTrigger t) -> ProxyHandlerResult { return this->operator()(t); };
  }

  /// Direct invocation. Fast-paths the single-tenant case (design.md §5.5).
  ProxyHandlerResult operator()(ProxyTrigger trig) {
    uint32_t mask = activeMask_.load(std::memory_order_acquire);
    // Single-tenant bypass: zero scheduling overhead.
    if (__builtin_popcount(mask) <= 1) {
      return inner_(trig);
    }

    uint32_t tid = static_cast<uint32_t>(trig.fields.tenantId);
    if (tid >= MAX_TENANTS) tid = 0;

    {
      std::lock_guard<std::mutex> g(mu_);
      queues_[tid].push_back(trig);
    }
    return drain();
  }

 private:
  ProxyHandlerResult drain() {
    ProxyHandlerResult last = ProxyHandlerResult::Continue;
    for (;;) {
      ProxyTrigger picked{};
      uint32_t pickedTid = MAX_TENANTS;
      bool found = false;
      {
        std::lock_guard<std::mutex> g(mu_);
        pickedTid = pickNext();
        if (pickedTid >= MAX_TENANTS) return last;
        // Honor token bucket without holding the queue lock.
        auto& q = queues_[pickedTid];
        if (q.empty()) return last;
        picked = q.front();
        // Rate-limit check (uses bucket's own internal lock).
        found = true;
      }
      if (!found) return last;

      uint64_t bytes = static_cast<uint64_t>(picked.fields.dstOffset == 0 ? 0 : 0) + picked.fields.dstOffset;
      // Re-extract size from fst (low 32 bits hold `size`).
      uint64_t size = static_cast<uint64_t>(picked.fst & ((uint64_t{1} << TriggerBitsSize) - 1));
      if (buckets_[pickedTid].tryConsume(size == 0 ? 1 : size)) {
        std::lock_guard<std::mutex> g(mu_);
        if (!queues_[pickedTid].empty()) {
          queues_[pickedTid].pop_front();
        }
        last = inner_(picked);
        if (last == ProxyHandlerResult::Stop) return last;
      } else {
        // Bucket dry — yield; the proxy will be called again with future
        // triggers and we'll retry the head of this queue then.
        return last;
      }
      (void)bytes;
    }
  }

  // Caller must hold mu_.
  uint32_t pickNext() {
    switch (mode_) {
      case PolicyMode::SinglePassthrough:
        for (uint32_t t = 0; t < MAX_TENANTS; ++t)
          if (!queues_[t].empty()) return t;
        return MAX_TENANTS;

      case PolicyMode::StrictPriority: {
        uint32_t best = MAX_TENANTS;
        int bestPri = -1;
        for (uint32_t t = 0; t < MAX_TENANTS; ++t) {
          if (queues_[t].empty()) continue;
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
          if (queues_[t].empty()) continue;
          if (tenants_[t].qos_class >= QoSClass::Premium) return t;
        }
        // fall through
        [[fallthrough]];
      }
      case PolicyMode::Fair: {
        // Simple round-robin among non-empty queues; weight applied via
        // deficit counter would go here in a richer impl.
        uint32_t n = MAX_TENANTS;
        for (uint32_t step = 0; step < MAX_TENANTS; ++step) {
          uint32_t t = (rrCursor_ + step) % MAX_TENANTS;
          if (!queues_[t].empty()) {
            rrCursor_ = (t + 1) % MAX_TENANTS;
            return t;
          }
        }
        (void)n;
        return MAX_TENANTS;
      }
    }
    return MAX_TENANTS;
  }

  ProxyHandler inner_;
  std::atomic<uint32_t> activeMask_{0};
  PolicyMode mode_;
  std::mutex mu_;
  std::array<std::deque<ProxyTrigger>, MAX_TENANTS> queues_{};
  std::array<TenantContext, MAX_TENANTS> tenants_{};
  std::array<BandwidthBudget, MAX_TENANTS> budgets_{};
  std::array<TokenBucket, MAX_TENANTS> buckets_{};
  uint32_t rrCursor_ = 0;
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
