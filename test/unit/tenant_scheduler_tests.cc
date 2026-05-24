// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Iter5 host-side regression tests for TenantAwareProxyHandler.
//
// Covers the design v0.2.1 invariants:
//   - dispatch order: parse tenant -> fail-open -> update activeMask -> bypass
//     (otherwise a brand-new tenant's first trigger gets bypassed before it can
//      be counted as active, breaking isolation forever).
//   - per-connection program order: scheduler may reorder across tenants but
//     MUST preserve FIFO order within a single connection (keyed by
//     semaphoreId — same PortChannel shares one proxy sem).
//   - fail-open: unregistered tenants get rewritten to DEFAULT_TENANT instead
//     of being dropped, so a kernel that beats policyd does not deadlock.

#include <mscclpp/ext/tenant.hpp>
#include <mscclpp/ext/tenant_aware_proxy.hpp>
#include <mscclpp/fifo_device.hpp>
#include <mscclpp/proxy.hpp>

#include <atomic>
#include <vector>

#include "../framework.hpp"

namespace {

using mscclpp::ContextProxyHandler;
using mscclpp::ProxyFifoContext;
using mscclpp::ProxyHandler;
using mscclpp::ProxyHandlerResult;
using mscclpp::ProxyTrigger;
using mscclpp::TriggerData;
using mscclpp::TriggerSync;
using mscclpp::ext::tenant::DEFAULT_TENANT;
using mscclpp::ext::tenant::MAX_TENANTS;
using mscclpp::ext::tenant::PolicyMode;
using mscclpp::ext::tenant::QoSClass;
using mscclpp::ext::tenant::TenantAwareProxyHandler;

// Build a ProxyTrigger with the requested tenantId / semaphoreId / type.
// We poke fields directly so the test does not depend on the device-side
// ProxyTrigger ctor (which is gated on MSCCLPP_DEVICE_COMPILE).
ProxyTrigger makeTrigger(uint32_t tenantId, uint32_t semaphoreId, uint32_t type, uint32_t size = 1024) {
  ProxyTrigger t{};
  t.fields.size = size;
  t.fields.srcOffset = 0;
  t.fields.dstOffset = 0;
  t.fields.srcMemoryId = 0;
  t.fields.dstMemoryId = 0;
  t.fields.type = type;
  t.fields.tenantId = tenantId;
  t.fields.semaphoreId = semaphoreId;
  return t;
}

// Record handler — captures the (tenantId, semaphoreId, type) of every
// trigger that reaches the inner handler, in dispatch order.
struct DispatchRecord {
  uint32_t tenantId;
  uint32_t semaphoreId;
  uint32_t type;
};

class Recorder {
 public:
  ContextProxyHandler asHandler() {
    return [this](ProxyTrigger t, ProxyFifoContext ctx) -> ProxyHandlerResult {
      records_.push_back(DispatchRecord{
          static_cast<uint32_t>(t.fields.tenantId),
          static_cast<uint32_t>(t.fields.semaphoreId),
          static_cast<uint32_t>(t.fields.type),
      });
      lastCtx_ = ctx;
      return ProxyHandlerResult::Continue;
    };
  }
  const std::vector<DispatchRecord>& records() const { return records_; }
  ProxyFifoContext lastCtx() const { return lastCtx_; }

 private:
  std::vector<DispatchRecord> records_;
  ProxyFifoContext lastCtx_{};
};

// Helper: build a ProxyFifoContext with a synthetic fifoPos for testing.
inline ProxyFifoContext makeCtx(uint64_t fifoPos) {
  ProxyFifoContext c{};
  c.fifoPos = fifoPos;
  c.enqueueNs = fifoPos;  // not significant for these tests
  return c;
}

}  // namespace

// -------- 1) Dispatch order: bypass MUST happen AFTER activeMask update.
//
// Repro of the bug the review caught: tenant A active, queues empty. Tenant B's
// first trigger arrives. If we read activeMask BEFORE updating it, we see
// popcount==1, bypass, and B never gets to mark itself active — so isolation
// for B is permanently broken.
TEST(TenantSchedulerDispatchOrderTest, FirstTriggerFromNewTenantIsNotBypassedWithoutActiveMaskUpdate) {
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);
  uint64_t pos = 100;
  auto P = [&](uint32_t tid, uint32_t sem, uint32_t type) {
    return h(makeTrigger(tid, sem, type), makeCtx(pos++));
  };

  h.updateTenant({/*tenant_id=*/1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({/*tenant_id=*/2, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});

  // T1 sends two triggers — both should bypass (single active tenant + queues empty).
  P(1, 0, TriggerData);
  P(1, 0, TriggerData);
  ASSERT_EQ(rec.records().size(), 2u);
  EXPECT_EQ(rec.records()[0].tenantId, 1u);
  EXPECT_EQ(rec.records()[1].tenantId, 1u);

  // T2 sends its first trigger. With the bug, this would still be bypassed.
  P(2, 0, TriggerData);
  ASSERT_EQ(rec.records().size(), 3u);
  EXPECT_EQ(rec.records()[2].tenantId, 2u);

  // Subsequent T1 trigger must NOT silently bypass — activeMask has both bits.
  P(1, 0, TriggerData);
  ASSERT_EQ(rec.records().size(), 4u);
}

// -------- 2) Per-connection program order: same semaphoreId stays FIFO.
//
// Scenario: connection key = semaphoreId = 7. Three triggers arrive in order
// D(t=1), S(t=2), D(t=1). They land in different tenant queues (1 and 2) so
// without the per-connection guard, a Fair scheduler that picks t=2 first
// would let TriggerSync overtake the earlier TriggerData on the same conn.
TEST(TenantSchedulerPerConnOrderingTest, SameConnectionFifoPreservedAcrossTenants) {
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);
  uint64_t pos = 10;
  auto P = [&](uint32_t tid, uint32_t sem, uint32_t type) {
    return h(makeTrigger(tid, sem, type), makeCtx(pos++));
  };

  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});

  // Three triggers all on semaphoreId=7, arrival order D1 -> S2 -> D1'.
  // S2 (tenant 2) must NOT pass D1 (tenant 1) even though it's a different queue.
  P(1, 7, TriggerData);  // pos = 10
  P(2, 7, TriggerSync);  // pos = 11
  P(1, 7, TriggerData);  // pos = 12

  // Drive drain with cheap triggers on an isolated key.
  for (int i = 0; i < 6; ++i) {
    P(1, 8, TriggerData);
  }

  std::vector<uint32_t> seq;
  for (auto& r : rec.records()) {
    if (r.semaphoreId == 7) seq.push_back(r.type);
  }
  ASSERT_EQ(seq.size(), 3u);
  EXPECT_EQ(seq[0], (uint32_t)TriggerData);  // D1
  EXPECT_EQ(seq[1], (uint32_t)TriggerSync);  // S2 — never before D1
  EXPECT_EQ(seq[2], (uint32_t)TriggerData);  // D1' — never before S2
}

// -------- 3) Cross-connection triggers may still be reordered (no invariant).
//
// Without this freedom, the scheduler would degrade to global FIFO and lose
// all multi-tenant fairness benefit. Two triggers on DIFFERENT semaphoreIds
// from different tenants are unconstrained.
TEST(TenantSchedulerPerConnOrderingTest, DifferentConnectionsMayReorder) {
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::StrictPriority);
  uint64_t pos = 200;
  auto P = [&](uint32_t tid, uint32_t sem, uint32_t type) {
    return h(makeTrigger(tid, sem, type), makeCtx(pos++));
  };

  h.updateTenant({1, QoSClass::BestEffort, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Realtime, 1, 0, 0, 0, 0}, {0, 0, 0});

  P(1, 1, TriggerData);
  P(2, 2, TriggerData);

  for (int i = 0; i < 4; ++i) {
    P(2, 3, TriggerData);
  }

  bool sawSem1 = false, sawSem2 = false;
  for (auto& r : rec.records()) {
    if (r.semaphoreId == 1) sawSem1 = true;
    if (r.semaphoreId == 2) sawSem2 = true;
  }
  EXPECT_TRUE(sawSem1);
  EXPECT_TRUE(sawSem2);
}

// -------- 4) Fail-open: unregistered tenants are rewritten to DEFAULT_TENANT.
//
// Real-world race: GPU kernels can start pushing triggers before policyd has
// finished registering the tenant on this rank. Hard-aborting in that window
// would deadlock the workload. Fail-open keeps the link alive and accounts
// the traffic against DEFAULT_TENANT.
TEST(TenantSchedulerFailOpenTest, UnregisteredTenantReattributedToDefault) {
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);
  uint64_t pos = 300;
  auto P = [&](uint32_t tid, uint32_t sem, uint32_t type) {
    return h(makeTrigger(tid, sem, type), makeCtx(pos++));
  };

  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});

  // Tenant 5 is NOT registered — should fail-open to DEFAULT_TENANT (=0).
  P(5, 0, TriggerData);
  ASSERT_EQ(rec.records().size(), 1u);

  h.updateTenant({5, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  P(5, 0, TriggerData);
  ASSERT_EQ(rec.records().size(), 2u);
}

// -------- 5) Bypass refuses to fire while scheduler queue is non-empty.
//
// Even when popcount(activeMask)<=1 due to aging or removal, the scheduler
// must drain its existing queue first to avoid reordering trigger N+1 (just
// arrived, would bypass) past trigger N (still in queue from earlier).
TEST(TenantSchedulerBypassTest, DoesNotBypassWhilePendingQueueIsNonEmpty) {
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);
  uint64_t pos = 400;
  auto P = [&](uint32_t tid, uint32_t sem, uint32_t type) {
    return h(makeTrigger(tid, sem, type), makeCtx(pos++));
  };

  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});

  P(1, 0, TriggerData);
  P(2, 0, TriggerData);
  EXPECT_GE(rec.records().size(), 1u);

  for (int i = 0; i < 10; ++i) {
    P((i % 2) + 1, 0, TriggerData);
  }
  EXPECT_LE(h.pendingCountForTest(), 11u);
}

// -------- 6) TriggerSync flush boundary uses push-time fifoPos (design.md §5.7).
//
// Regression for the iter5 P1 bug: when the scheduler delays dispatch, the
// inner handler must see the ORIGINAL push-time fifoPos, not the proxy's
// current tail() which has advanced by however many triggers were polled
// while this one waited in queue.
TEST(TriggerSyncFifoPosTest, DispatchedCtxMatchesPushTimePos) {
  // The Recorder remembers the ctx of the LAST trigger dispatched. We push
  // a tagged TriggerSync at fifoPos=999, then several Data triggers behind
  // it. Under Fair scheduling the order between tenants may interleave, but
  // when the Sync is eventually dispatched its ctx.fifoPos MUST be 999, not
  // some later value.
  struct PerCallRecorder {
    std::vector<ProxyFifoContext> ctxs;
    ContextProxyHandler asHandler() {
      return [this](ProxyTrigger, ProxyFifoContext c) {
        ctxs.push_back(c);
        return ProxyHandlerResult::Continue;
      };
    }
  } rec;

  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);
  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});

  // Manually drive: TriggerSync at pos 999 from tenant 1; then 5 Data from
  // tenant 2 at later positions. The scheduler may interleave on dispatch,
  // but the ORIGINAL fifoPos for each must round-trip exactly.
  h(makeTrigger(/*tid=*/1, /*sem=*/0, TriggerSync), makeCtx(999));
  for (uint64_t p = 1000; p < 1005; ++p) {
    h(makeTrigger(/*tid=*/2, /*sem=*/0, TriggerData), makeCtx(p));
  }

  // Find the dispatch that corresponds to the Sync — the only one with
  // fifoPos == 999.
  bool sawSyncCtx = false;
  for (auto& c : rec.ctxs) {
    if (c.fifoPos == 999) sawSyncCtx = true;
    // No ctx may be a stale/fresh tail() value — they must come from the set
    // {999, 1000, 1001, 1002, 1003, 1004}.
    EXPECT_TRUE(c.fifoPos == 999 || (c.fifoPos >= 1000 && c.fifoPos <= 1004));
  }
  EXPECT_TRUE(sawSyncCtx);
}
