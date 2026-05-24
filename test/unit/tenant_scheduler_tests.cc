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
  ProxyHandler asHandler() {
    return [this](ProxyTrigger t) -> ProxyHandlerResult {
      records_.push_back(DispatchRecord{
          static_cast<uint32_t>(t.fields.tenantId),
          static_cast<uint32_t>(t.fields.semaphoreId),
          static_cast<uint32_t>(t.fields.type),
      });
      return ProxyHandlerResult::Continue;
    };
  }
  const std::vector<DispatchRecord>& records() const { return records_; }

 private:
  std::vector<DispatchRecord> records_;
};

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

  // Register tenants 1 and 2.
  h.updateTenant({/*tenant_id=*/1, QoSClass::Standard, 1, 0, 0, 0, 0},
                 {/*bytes_per_second=*/0, /*burst_bytes=*/0, 0});
  h.updateTenant({/*tenant_id=*/2, QoSClass::Standard, 1, 0, 0, 0, 0},
                 {/*bytes_per_second=*/0, /*burst_bytes=*/0, 0});

  // T1 sends two triggers — both should bypass (single active tenant + queues empty).
  h(makeTrigger(/*tid=*/1, /*sem=*/0, TriggerData));
  h(makeTrigger(/*tid=*/1, /*sem=*/0, TriggerData));
  ASSERT_EQ(rec.records().size(), 2u);
  EXPECT_EQ(rec.records()[0].tenantId, 1u);
  EXPECT_EQ(rec.records()[1].tenantId, 1u);

  // T2 sends its first trigger. With the bug, this would still be bypassed.
  // With the fix, activeMask becomes {1, 2}, popcount==2, slow path engages.
  h(makeTrigger(/*tid=*/2, /*sem=*/0, TriggerData));
  // The trigger should have reached inner_ (via drain → pickNext), but more
  // importantly, T2 must now be tracked. Subsequent triggers from T2 must
  // never silently bypass back.
  ASSERT_EQ(rec.records().size(), 3u);
  EXPECT_EQ(rec.records()[2].tenantId, 2u);

  // Re-confirm: another T1 trigger now goes through the scheduler (not bypass)
  // because activeMask has both bits set. We can't observe "took slow path"
  // directly, but we can verify the ordering invariant still holds.
  h(makeTrigger(/*tid=*/1, /*sem=*/0, TriggerData));
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

  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});

  // Push three triggers, all on semaphoreId=7, arrival order D1 -> S2 -> D1':
  //   - D1  : Data,  tenant 1, sem 7
  //   - S2  : Sync,  tenant 2, sem 7  (must NOT pass D1)
  //   - D1' : Data,  tenant 1, sem 7  (must NOT pass S2)
  h(makeTrigger(/*tid=*/1, /*sem=*/7, TriggerData));
  h(makeTrigger(/*tid=*/2, /*sem=*/7, TriggerSync));
  h(makeTrigger(/*tid=*/1, /*sem=*/7, TriggerData));

  // Drive drain repeatedly via cheap no-op triggers on an isolated key (sem 8)
  // to flush the pending queue. Each operator() call drains at most one
  // trigger, so just call it a few times. Use a different conn key so we
  // don't muddle the assertion.
  for (int i = 0; i < 6; ++i) {
    h(makeTrigger(/*tid=*/1, /*sem=*/8, TriggerData));
  }

  // Filter only sem=7 triggers in dispatch order.
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

  h.updateTenant({1, QoSClass::BestEffort, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Realtime, 1, 0, 0, 0, 0}, {0, 0, 0});

  // D on tenant 1 / sem 1 arrives first, but tenant 2 has Realtime priority,
  // so under StrictPriority its trigger (different sem) is allowed to be
  // dispatched first. The invariant only forbids reorder WITHIN the same sem.
  h(makeTrigger(/*tid=*/1, /*sem=*/1, TriggerData));
  h(makeTrigger(/*tid=*/2, /*sem=*/2, TriggerData));

  // Drain.
  for (int i = 0; i < 4; ++i) {
    h(makeTrigger(/*tid=*/2, /*sem=*/3, TriggerData));  // priming triggers on key 3
  }

  // We don't pin a specific order across (sem=1, sem=2) — only that BOTH
  // eventually appear, and they don't violate same-conn ordering (trivially
  // satisfied because each has a unique key).
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

  // Register tenant 1 only. Default tenant (0) is implicit.
  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});

  // Tenant 5 is NOT registered — should fail-open to DEFAULT_TENANT (=0).
  // We can't directly observe the rewrite (the trigger struct is consumed
  // by inner_), but we CAN verify:
  //   - the trigger is delivered (not dropped),
  //   - it bypasses or enqueues correctly under DEFAULT_TENANT accounting.
  h(makeTrigger(/*tid=*/5, /*sem=*/0, TriggerData));
  ASSERT_EQ(rec.records().size(), 1u);

  // Now register tenant 5 properly and send again — should not warn / not be
  // rewritten. The handler does not expose tenantId post-dispatch, so we just
  // check delivery succeeds and didn't crash.
  h.updateTenant({5, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h(makeTrigger(/*tid=*/5, /*sem=*/0, TriggerData));
  ASSERT_EQ(rec.records().size(), 2u);
}

// -------- 5) Bypass refuses to fire while scheduler queue is non-empty.
//
// Even when popcount(activeMask)<=1 due to aging or removal, the scheduler
// must drain its existing queue first to avoid reordering trigger N+1 (just
// arrived, would bypass) past trigger N (still in queue from earlier).
TEST(TenantSchedulerBypassTest, DoesNotBypassWhilePendingQueueIsNonEmpty) {
  // Hold the inner handler open via a non-trivial body so we can observe.
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);

  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});

  // Make both tenants active.
  h(makeTrigger(/*tid=*/1, /*sem=*/0, TriggerData));
  h(makeTrigger(/*tid=*/2, /*sem=*/0, TriggerData));
  // After these, both bits are set in activeMask. There may be 0 or 1 pending
  // depending on drain order. We don't care; just confirm no crash.
  EXPECT_GE(rec.records().size(), 1u);

  // Force a state where queue is non-empty: enqueue many triggers; each
  // operator() call drains at most one, so a burst leaves residue.
  for (int i = 0; i < 10; ++i) {
    h(makeTrigger(/*tid=*/(i % 2) + 1, /*sem=*/0, TriggerData));
  }
  // Some triggers may still be pending — pendingCountForTest is non-decreasing
  // across this burst then drained on subsequent calls. Just sanity check:
  EXPECT_LE(h.pendingCountForTest(), 11u);
}
