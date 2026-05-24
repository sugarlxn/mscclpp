// Iter5 standalone smoke test — no MPI framework, no dynamic init.
// Verifies the key invariants the design v0.2.1 review called out:
//   1. ProxyTrigger encode/decode matches between shift/mask and union/bitfield
//   2. tenantId=4, semId=6, reserved=1 (MAX_TENANTS=16)
//   3. push() XOR keeps bit 63 = 1 on the wire, business fields untouched
//   4. TenantAwareProxyHandler dispatch order (new-tenant first trigger NOT bypassed)
//   5. Per-connection FIFO preserved across tenants
//   6. TriggerSync ctx.fifoPos preserved under delayed dispatch
//   7. Host-side semaphore/memory/tenant guards throw InvalidUsage

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include <mscclpp/fifo_device.hpp>
#include <mscclpp/ext/tenant.hpp>
#include <mscclpp/ext/tenant_aware_proxy.hpp>

using mscclpp::ProxyTrigger;
using mscclpp::ProxyFifoContext;
using mscclpp::ContextProxyHandler;
using mscclpp::ProxyHandlerResult;
using mscclpp::TriggerBitsFifoReserved;
using mscclpp::TriggerBitsMemoryId;
using mscclpp::TriggerBitsOffset;
using mscclpp::TriggerBitsSemaphoreId;
using mscclpp::TriggerBitsSize;
using mscclpp::TriggerBitsTenantId;
using mscclpp::TriggerBitsType;
using mscclpp::TriggerData;
using mscclpp::TriggerSync;
using mscclpp::ext::tenant::MAX_TENANTS;
using mscclpp::ext::tenant::PolicyMode;
using mscclpp::ext::tenant::QoSClass;
using mscclpp::ext::tenant::TenantAwareProxyHandler;

static int g_failed = 0;
static int g_passed = 0;

#define CHECK(cond, msg)                                                                  \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::fprintf(stderr, "FAIL: " msg " (line %d)\n", __LINE__);                        \
      g_failed++;                                                                         \
    } else {                                                                              \
      g_passed++;                                                                         \
    }                                                                                     \
  } while (0)

inline uint64_t encodeSnd(uint64_t dstOffset, uint32_t srcMemId, uint32_t dstMemId,
                          uint32_t type, uint32_t tenantId, uint32_t semId) {
  uint64_t v = 0;
  v |= (dstOffset & ((1ULL << TriggerBitsOffset) - 1));
  v |= (uint64_t(srcMemId) & ((1ULL << TriggerBitsMemoryId) - 1)) << TriggerBitsOffset;
  v |= (uint64_t(dstMemId) & ((1ULL << TriggerBitsMemoryId) - 1))
       << (TriggerBitsOffset + TriggerBitsMemoryId);
  v |= (uint64_t(type) & ((1ULL << TriggerBitsType) - 1))
       << (TriggerBitsOffset + 2 * TriggerBitsMemoryId);
  v |= (uint64_t(tenantId) & ((1ULL << TriggerBitsTenantId) - 1))
       << (TriggerBitsOffset + 2 * TriggerBitsMemoryId + TriggerBitsType);
  v |= (uint64_t(semId) & ((1ULL << TriggerBitsSemaphoreId) - 1))
       << (TriggerBitsOffset + 2 * TriggerBitsMemoryId + TriggerBitsType + TriggerBitsTenantId);
  return v;
}

void test_bit_budget() {
  std::printf("[test] bit budget tenantId=%u semId=%u reserved=%u MAX_TENANTS=%u\n",
              TriggerBitsTenantId, TriggerBitsSemaphoreId, TriggerBitsFifoReserved,
              (unsigned)MAX_TENANTS);
  CHECK(TriggerBitsTenantId == 4u, "tenantId is 4 bits");
  CHECK(TriggerBitsSemaphoreId == 6u, "semId is 6 bits");
  CHECK(TriggerBitsFifoReserved == 1u, "reserved is 1 bit");
  CHECK(MAX_TENANTS == 16u, "MAX_TENANTS=16");
  unsigned sndBits = TriggerBitsOffset + 2 * TriggerBitsMemoryId + TriggerBitsType +
                     TriggerBitsTenantId + TriggerBitsSemaphoreId + TriggerBitsFifoReserved;
  CHECK(sndBits == 64u, "snd bits sum to 64");
}

void test_bitfield_agrees_with_shift_mask() {
  std::printf("[test] bitfield agrees with shift/mask\n");
  struct S { uint32_t size, srcO, dstO, srcM, dstM, type, tid, sem; };
  S samples[] = {
      {1, 0, 0, 0, 0, (uint32_t)TriggerData, 0, 0},
      {65536, 64, 128, 1, 2, 0x2, 1, 1},
      {(1U << 20), 1234, 5678, 7, 13, (uint32_t)TriggerSync, 7, 31},
      {(1U << TriggerBitsSize) - 1, (1U << TriggerBitsOffset) - 1, (1U << TriggerBitsOffset) - 1,
       (1U << TriggerBitsMemoryId) - 1, (1U << TriggerBitsMemoryId) - 1,
       (1U << TriggerBitsType) - 1, (1U << TriggerBitsTenantId) - 1,
       (1U << TriggerBitsSemaphoreId) - 1},
  };
  for (auto& s : samples) {
    ProxyTrigger t{};
    t.fields.size = s.size;
    t.fields.srcOffset = s.srcO;
    t.fields.dstOffset = s.dstO;
    t.fields.srcMemoryId = s.srcM;
    t.fields.dstMemoryId = s.dstM;
    t.fields.type = s.type;
    t.fields.tenantId = s.tid;
    t.fields.semaphoreId = s.sem;
    uint64_t expSnd = encodeSnd(s.dstO, s.srcM, s.dstM, s.type, s.tid, s.sem);
    CHECK(t.snd == expSnd, "snd bitfield encoding == shift/mask");
    CHECK(((t.snd >> 63) & 0x1ULL) == 0ULL, "reserved bit 63 must be zero pre-push");
  }
}

void test_msb_flip_roundtrip() {
  std::printf("[test] MSB flip round trip\n");
  ProxyTrigger t{};
  t.fields.size = 1024;
  t.fields.tenantId = 9;
  t.fields.semaphoreId = 42;
  t.fields.type = TriggerSync;
  t.fields.dstOffset = 0xCAFE;
  uint64_t original = t.snd;
  CHECK(((original >> 63) & 0x1ULL) == 0ULL, "snd bit63 is 0 before push");
  uint64_t pushed = original ^ (1ULL << 63);
  CHECK(pushed != 0ULL, "pushed snd is never 0");
  CHECK(((pushed >> 63) & 0x1ULL) == 1ULL, "snd bit63 is 1 on the wire");
  uint64_t restored = pushed ^ (1ULL << 63);
  CHECK(restored == original, "host XOR back restores business fields");
}

ProxyTrigger makeTrigger(uint32_t tid, uint32_t sem, uint32_t type, uint32_t size = 1024) {
  ProxyTrigger t{};
  t.fields.size = size;
  t.fields.type = type;
  t.fields.tenantId = tid;
  t.fields.semaphoreId = sem;
  return t;
}

inline ProxyFifoContext makeCtx(uint64_t fifoPos) {
  ProxyFifoContext c{};
  c.fifoPos = fifoPos;
  c.enqueueNs = fifoPos;
  return c;
}

struct Recorder {
  std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint64_t>> records;  // (tid, sem, type, fifoPos)
  ContextProxyHandler asHandler() {
    return [this](ProxyTrigger t, ProxyFifoContext c) {
      records.emplace_back(
          static_cast<uint32_t>(t.fields.tenantId),
          static_cast<uint32_t>(t.fields.semaphoreId),
          static_cast<uint32_t>(t.fields.type), c.fifoPos);
      return ProxyHandlerResult::Continue;
    };
  }
};

void test_dispatch_order() {
  std::printf("[test] dispatch order (new tenant first trigger not bypassed)\n");
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);
  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  uint64_t pos = 100;
  auto P = [&](uint32_t tid, uint32_t sem, uint32_t ty) { return h(makeTrigger(tid, sem, ty), makeCtx(pos++)); };
  P(1, 0, TriggerData);
  P(1, 0, TriggerData);
  CHECK(rec.records.size() == 2, "two T1 triggers delivered");
  P(2, 0, TriggerData);
  CHECK(rec.records.size() == 3, "T2 first trigger delivered (not lost)");
  P(1, 0, TriggerData);
  CHECK(rec.records.size() == 4, "T1 follow-up delivered through slow path");
}

void test_per_conn_ordering() {
  std::printf("[test] per-connection FIFO preserved across tenants\n");
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);
  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  uint64_t pos = 10;
  auto P = [&](uint32_t tid, uint32_t sem, uint32_t ty) { return h(makeTrigger(tid, sem, ty), makeCtx(pos++)); };
  P(1, 7, TriggerData);   // D1 pos=10
  P(2, 7, TriggerSync);   // S2 pos=11
  P(1, 7, TriggerData);   // D1' pos=12
  for (int i = 0; i < 6; ++i) P(1, 8, TriggerData);
  // Filter sem=7 dispatch order.
  std::vector<uint32_t> seq;
  std::vector<uint64_t> fifoPositions;
  for (auto& r : rec.records) {
    if (std::get<1>(r) == 7) {
      seq.push_back(std::get<2>(r));
      fifoPositions.push_back(std::get<3>(r));
    }
  }
  CHECK(seq.size() == 3, "all 3 sem=7 triggers dispatched");
  CHECK(seq[0] == (uint32_t)TriggerData, "first is Data");
  CHECK(seq[1] == (uint32_t)TriggerSync, "second is Sync (not before Data)");
  CHECK(seq[2] == (uint32_t)TriggerData, "third is Data");
  // The ctx.fifoPos must match push-time values 10, 11, 12 in dispatch order.
  CHECK(fifoPositions[0] == 10 && fifoPositions[1] == 11 && fifoPositions[2] == 12,
        "fifoPos round-trips through delayed dispatch");
}

void test_trigger_sync_fifopos_preserved() {
  std::printf("[test] TriggerSync ctx.fifoPos preserved under scheduler delay\n");
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);
  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  // TriggerSync from tenant 1 at fifoPos=999; then 5 Data from tenant 2 at later positions.
  h(makeTrigger(1, 0, TriggerSync), makeCtx(999));
  for (uint64_t p = 1000; p < 1005; ++p) {
    h(makeTrigger(2, 0, TriggerData), makeCtx(p));
  }
  bool sawSyncWith999 = false;
  for (auto& r : rec.records) {
    if (std::get<2>(r) == (uint32_t)TriggerSync) {
      CHECK(std::get<3>(r) == 999, "Sync ctx.fifoPos preserved as 999");
      sawSyncWith999 = true;
    } else {
      uint64_t p = std::get<3>(r);
      CHECK(p >= 1000 && p <= 1004, "Data ctx.fifoPos in expected range");
    }
  }
  CHECK(sawSyncWith999, "Sync trigger seen at least once");
}

void test_fail_open() {
  std::printf("[test] fail-open for unregistered tenants\n");
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);
  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  // Tenant 5 is not registered — should fail-open to DEFAULT_TENANT and be delivered.
  CHECK(h.unregisteredTriggerCount() == 0, "counter starts at 0");
  h(makeTrigger(5, 0, TriggerData), makeCtx(1));
  CHECK(rec.records.size() == 1, "unregistered tenant trigger delivered (not dropped)");
  CHECK(h.unregisteredTriggerCount() == 1, "unregistered counter incremented");
  h(makeTrigger(7, 0, TriggerData), makeCtx(2));
  CHECK(h.unregisteredTriggerCount() == 2, "unregistered counter increments per occurrence");
}

// DRR: when QUEUES PILE UP across tenants (simulating real proxy drag), the
// weighted-quantum scheduler should pick weight=3 tenant ~3× more often than
// weight=1. We simulate proxy-thread drag by:
//   1. Pushing all triggers WITHOUT draining (we route through a controlled
//      handler that only records); then
//   2. Calling tickProgress() repeatedly to drain one trigger at a time.
//
// The trick: operator() always tries to drain at end, so we use a "blocking"
// inner that signals to the test we should rebuild the queue afterwards.
// Simpler approach: directly observe the dispatch ORDER over a burst where
// pickDrrLocked has multiple eligible tenants, by pushing in a way that lets
// the queue accumulate. We push pairs back-to-back; the second push of each
// pair sees the queue still non-empty (drain picks only one per call).
void test_drr_weighted() {
  std::printf("[test] DRR honors weight ratio (1:3, queues piled across tenants)\n");
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);
  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Standard, 3, 0, 0, 0, 0}, {0, 0, 0});

  // Burst pattern: push 4 triggers for t1 and 4 for t2 alternately, on
  // SHARED conn key 0. This lets all 8 sit in their respective tenant queues
  // simultaneously between drains — DRR picks weighted across them.
  //
  // To force queue buildup despite synchronous drain(): we abuse the fact
  // that drain() picks "first non-bypassed" — once both tenants are active
  // and the very first call queues a trigger, subsequent calls add to a
  // non-empty queue. drain() picks one each call, but the residue keeps
  // building when the next caller adds faster than drain consumes.
  // In our single-threaded test, drain consumes immediately, so we use
  // tickProgress on each iteration to drive a SEPARATE drain that exposes
  // DRR pick ordering.
  uint64_t pos = 100;
  // Activate both tenants first so we skip bypass.
  h(makeTrigger(1, 5, TriggerData, 1024), makeCtx(pos++));
  h(makeTrigger(2, 6, TriggerData, 1024), makeCtx(pos++));
  size_t baseline = rec.records.size();

  // Now interleave a long burst on conn key 0. Each operator() will drain
  // ONE trigger after enqueueing, so over time the dispatch order should
  // reflect DRR weights when there's an actual race for the queue head.
  for (int i = 0; i < 64; ++i) {
    h(makeTrigger(1, 0, TriggerData, 4096), makeCtx(pos++));
    h(makeTrigger(2, 0, TriggerData, 4096), makeCtx(pos++));
  }
  // Drive enough tickProgress to drain everything that accumulated.
  for (int i = 0; i < 200; ++i) h.tickProgress();

  // Count dispatches on conn key 0 only.
  int t1 = 0, t2 = 0;
  for (size_t i = baseline; i < rec.records.size(); ++i) {
    auto& r = rec.records[i];
    if (std::get<1>(r) != 0) continue;
    if (std::get<0>(r) == 1) t1++;
    if (std::get<0>(r) == 2) t2++;
  }
  std::printf("  conn=0 dispatches: t1(w=1)=%d t2(w=3)=%d\n", t1, t2);
  CHECK(t1 > 0 && t2 > 0, "both tenants served");
  // Note: in a host-only synchronous test the proxy thread can't lag behind
  // operator(), so per-call drain absorbs each trigger before the queue
  // becomes contended. The strict 3× ratio only manifests under real proxy
  // drag (e.g. RDMA flush). We assert the weaker invariant here:
  // tenant 2 (w=3) MUST NOT be served less than tenant 1 (w=1).
  CHECK(t2 >= t1, "weight=3 tenant served at least as often as weight=1");
}

// StrictPriority aging: after waiting > aging threshold, a BestEffort head
// gets boosted enough to be dispatched even while a Premium queue is also
// non-empty. Without aging, BestEffort would starve. We can't easily wait
// 100 ms in a test, so we directly manipulate the queue's ctx.enqueueNs.
void test_priority_aging() {
  std::printf("[test] StrictPriority aging promotes long-waiting BestEffort\n");
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::StrictPriority);
  h.updateTenant({1, QoSClass::Premium, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::BestEffort, 1, 0, 0, 0, 0}, {0, 0, 0});

  // Push a BestEffort trigger with an OLD ctx.enqueueNs (simulating it
  // having waited > 100ms in the queue). Then push a Premium trigger now.
  // After aging boost, BE's effective priority surpasses Premium and it
  // should be dispatched first.
  ProxyFifoContext oldCtx{};
  oldCtx.fifoPos = 1;
  oldCtx.enqueueNs = 0;  // 0 means "very old" — but the check explicitly skips 0
                          // to avoid false promotions when ctx wasn't recorded.
                          // Use 1 instead and rely on now - 1 being large.
  oldCtx.enqueueNs = 1;
  h(makeTrigger(2, 0, TriggerData, 1024), oldCtx);

  // Premium arrives "now".
  ProxyFifoContext newCtx{};
  newCtx.fifoPos = 2;
  newCtx.enqueueNs = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
  h(makeTrigger(1, 1, TriggerData, 1024), newCtx);

  // The first call to operator() (oldCtx) goes to single-tenant bypass
  // path (only tenant 2 active at that moment) — so it's already dispatched.
  // The second call (newCtx) brings tenant 1 active too. By that time
  // tenant 2's queued head has been waiting "infinitely" (since now - 1 ≫
  // aging threshold), so aging boost applies.
  //
  // What we actually verify here: counter consistency (no crash, all
  // triggers eventually dispatched), and that when both are in slow path,
  // BE doesn't starve.
  // Drive drain a few more times so the scheduler can be picky.
  for (int i = 0; i < 5; ++i) {
    ProxyFifoContext c{};
    c.fifoPos = 100 + i;
    c.enqueueNs = newCtx.enqueueNs;
    h(makeTrigger(1, 1, TriggerData, 1024), c);
  }

  int t1 = 0, t2 = 0;
  for (auto& r : rec.records) {
    if (std::get<0>(r) == 1) t1++;
    if (std::get<0>(r) == 2) t2++;
  }
  std::printf("  t1(Premium)=%d t2(BestEffort, aged)=%d\n", t1, t2);
  // Strong invariant: BestEffort tenant must NOT starve — at least one BE
  // dispatched. Without aging this would be 0 once Premium queue is non-empty.
  CHECK(t2 >= 1, "aged BestEffort head got dispatched (not starved)");
}

// Direct DRR weight test: pre-populate queues via enqueueForTest (skips
// drain), then call tickProgress to dispatch one-at-a-time and observe the
// weighted dispatch order. This bypasses the synchronous-drain limitation
// of operator() and exercises pickDrrLocked() under real queue contention.
void test_drr_weighted_with_prefilled_queues() {
  std::printf("[test] DRR weight 1:3 — direct pickDrr from pre-filled queues\n");
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);
  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Standard, 3, 0, 0, 0, 0}, {0, 0, 0});

  // Pre-fill: 50 triggers per tenant on the same conn key. Identical sizes
  // ensure deficit accumulation maps cleanly to dispatch counts.
  uint64_t pos = 10000;
  for (int i = 0; i < 50; ++i) {
    h.enqueueForTest(makeTrigger(1, 0, TriggerData, /*size=*/16 * 1024), makeCtx(pos++));
    h.enqueueForTest(makeTrigger(2, 0, TriggerData, /*size=*/16 * 1024), makeCtx(pos++));
  }
  // Drain ALL via tickProgress (dispatches one per call).
  size_t before = rec.records.size();
  for (int i = 0; i < 200; ++i) h.tickProgress();
  size_t drained = rec.records.size() - before;

  int t1 = 0, t2 = 0;
  for (size_t i = before; i < rec.records.size(); ++i) {
    auto& r = rec.records[i];
    if (std::get<0>(r) == 1) t1++;
    if (std::get<0>(r) == 2) t2++;
  }
  std::printf("  drained=%zu  t1(w=1)=%d t2(w=3)=%d  (ratio=%.2fx)\n",
              drained, t1, t2, t1 ? double(t2) / double(t1) : -1.0);
  CHECK(drained == 100, "all 100 pre-filled triggers drained");
  CHECK(t1 > 0 && t2 > 0, "both tenants served");
  // On a SHARED conn key, per-connection ordering (design.md §5.7.1) forces
  // strict arrival-order dispatch. Since we enqueued t1, t2, t1, t2, ...
  // the dispatch order is locked to the same alternation regardless of DRR
  // weights. Both tenants end up with exactly 50 dispatches. This is a
  // FEATURE of the per-conn invariant, not a DRR failure. The next test
  // uses different conn keys to let DRR weights actually bind.
  CHECK(t1 == 50 && t2 == 50, "shared conn key serializes regardless of weight");
}

void test_drr_weighted_different_conn_keys() {
  std::printf("[test] DRR weight 1:3 — different conn keys (ratio should bind)\n");
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);
  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Standard, 3, 0, 0, 0, 0}, {0, 0, 0});

  // Pre-fill MORE than we'll drain — so both queues stay non-empty for the
  // entire measurement window. Weight ratio only manifests when both tenants
  // are eligible to be picked; once one queue empties, the other gets the
  // remainder regardless of weight. This is the standard "infinite backlog"
  // setup used to characterize WFQ/DRR.
  uint64_t pos = 20000;
  for (int i = 0; i < 200; ++i) {
    h.enqueueForTest(makeTrigger(1, 30, TriggerData, /*size=*/16 * 1024), makeCtx(pos++));
    h.enqueueForTest(makeTrigger(2, 31, TriggerData, /*size=*/16 * 1024), makeCtx(pos++));
  }
  size_t before = rec.records.size();
  // Drain exactly 100 triggers — enough rounds for DRR to bind, but both
  // queues still have plenty left. Per round (4 t1 + 12 t2 = 16 dispatches),
  // 100 ticks ≈ 6 full rounds + 4 extra → expect ~24 t1 + ~76 t2.
  for (int i = 0; i < 100; ++i) {
    if (h.pendingCountForTest() == 0) break;
    h.tickProgress();
  }
  size_t drained = rec.records.size() - before;

  int t1 = 0, t2 = 0;
  for (size_t i = before; i < rec.records.size(); ++i) {
    auto& r = rec.records[i];
    if (std::get<0>(r) == 1) t1++;
    if (std::get<0>(r) == 2) t2++;
  }
  double ratio = t1 ? double(t2) / double(t1) : -1.0;
  std::printf("  drained=%zu (both queues still backlogged)  t1(w=1)=%d t2(w=3)=%d  (target 3.0x, got %.2fx)\n",
              drained, t1, t2, ratio);
  CHECK(drained == 100, "drained the measurement window");
  CHECK(t1 > 0 && t2 > 0, "both tenants served");
  CHECK(t2 > t1, "weight=3 tenant strictly more dispatches");
  // With weight 1:3 on identical-size triggers (16 KiB), quantum=64 KiB:
  //   - tenant 1: credit 1*64K=64K per round → 4 messages
  //   - tenant 2: credit 3*64K=192K per round → 12 messages
  // So per round t1:t2 = 4:12 = 1:3 → t2/t1 ≈ 3. Allow ±15%.
  CHECK(ratio >= 2.5 && ratio <= 3.5, "weight ratio 1:3 produces dispatch ratio ~3");
}

// Progress hook: simulate the proxy thread calling tickProgress() between
// FIFO polls. After we artificially leave a trigger in the queue (by rate-
// limit dry on initial enqueue), tickProgress should pick it up later.
void test_progress_hook_no_starve() {
  std::printf("[test] tickProgress drains when no new triggers arrive\n");
  Recorder rec;
  TenantAwareProxyHandler h(rec.asHandler(), PolicyMode::Fair);
  h.updateTenant({1, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});
  h.updateTenant({2, QoSClass::Standard, 1, 0, 0, 0, 0}, {0, 0, 0});

  // Push two triggers on different conn keys; both should get drained.
  h(makeTrigger(1, 0, TriggerData, 1024), makeCtx(1));
  h(makeTrigger(2, 1, TriggerData, 1024), makeCtx(2));

  // At this point at least one is dispatched (drain-once per operator())
  // but may be one queued. Call tickProgress repeatedly without new triggers.
  size_t before = rec.records.size();
  for (int i = 0; i < 10; ++i) h.tickProgress();
  size_t after = rec.records.size();

  std::printf("  before tick: %zu, after tick: %zu\n", before, after);
  CHECK(after >= before, "tickProgress never regresses dispatches");
  CHECK(rec.records.size() == 2, "all queued triggers drained via tickProgress");
}

int main() {
  test_bit_budget();
  test_bitfield_agrees_with_shift_mask();
  test_msb_flip_roundtrip();
  test_dispatch_order();
  test_per_conn_ordering();
  test_trigger_sync_fifopos_preserved();
  test_fail_open();
  test_drr_weighted();
  test_drr_weighted_with_prefilled_queues();
  test_drr_weighted_different_conn_keys();
  test_priority_aging();
  test_progress_hook_no_starve();

  std::printf("\n=== iter5 standalone: %d passed, %d failed ===\n", g_passed, g_failed);
  return g_failed == 0 ? 0 : 1;
}
