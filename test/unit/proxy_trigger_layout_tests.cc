// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// MT-MSCCL++ design v0.2.1 (§5.2): ProxyTrigger ABI test using **explicit
// shift/mask** encode/decode as the source of truth, cross-checked against
// the C++ `union { uint64_t; struct { … : bits; }; }` representation.
//
// Why this test exists: bitfield layout is compiler-dependent in principle
// (GCC/Clang on x86_64 happens to match our shift/mask, but MSVC has been
// known to differ on field packing across byte boundaries). When the on-wire
// ProxyTrigger needs to be parsed by both device-side (CUDA) and host-side
// (proxy thread) code, both ends must agree byte-for-byte. The shift/mask
// helpers in this file are the canonical reference; the union/struct path
// is convenience syntax that MUST produce the same 128 bits.

#include <mscclpp/fifo_device.hpp>

#include <cstdint>

#include "../framework.hpp"

namespace {

using mscclpp::ProxyTrigger;
using mscclpp::TriggerBitsFifoReserved;
using mscclpp::TriggerBitsMemoryId;
using mscclpp::TriggerBitsOffset;
using mscclpp::TriggerBitsSemaphoreId;
using mscclpp::TriggerBitsSize;
using mscclpp::TriggerBitsTenantId;
using mscclpp::TriggerBitsType;
using mscclpp::TriggerData;
using mscclpp::TriggerFlag;
using mscclpp::TriggerSync;

// Canonical shift/mask encoder for ProxyTrigger.snd. Layout (low → high):
//   dstOffset(32) | srcMemId(9) | dstMemId(9) | type(3) | tenantId(4) | semId(6) | reserved(1)
// reserved MUST be 0 on the wire (push() XORs bit 63 to ensure snd != 0).
inline uint64_t encodeFst(uint64_t size, uint64_t srcOffset) {
  constexpr uint64_t maskSize = (1ULL << TriggerBitsSize) - 1;
  constexpr uint64_t maskOffset = (1ULL << TriggerBitsOffset) - 1;
  return (size & maskSize) | ((srcOffset & maskOffset) << TriggerBitsSize);
}

inline uint64_t encodeSnd(uint64_t dstOffset, uint32_t srcMemId, uint32_t dstMemId,
                          uint32_t type, uint32_t tenantId, uint32_t semId) {
  constexpr uint64_t maskOffset = (1ULL << TriggerBitsOffset) - 1;
  constexpr uint64_t maskMemId = (1ULL << TriggerBitsMemoryId) - 1;
  constexpr uint64_t maskType = (1ULL << TriggerBitsType) - 1;
  constexpr uint64_t maskTenantId = (1ULL << TriggerBitsTenantId) - 1;
  constexpr uint64_t maskSemId = (1ULL << TriggerBitsSemaphoreId) - 1;

  uint64_t v = 0;
  v |= (dstOffset & maskOffset);
  v |= (uint64_t(srcMemId) & maskMemId) << (TriggerBitsOffset);
  v |= (uint64_t(dstMemId) & maskMemId) << (TriggerBitsOffset + TriggerBitsMemoryId);
  v |= (uint64_t(type) & maskType) << (TriggerBitsOffset + 2 * TriggerBitsMemoryId);
  v |= (uint64_t(tenantId) & maskTenantId)
       << (TriggerBitsOffset + 2 * TriggerBitsMemoryId + TriggerBitsType);
  v |= (uint64_t(semId) & maskSemId)
       << (TriggerBitsOffset + 2 * TriggerBitsMemoryId + TriggerBitsType + TriggerBitsTenantId);
  // reserved bit stays 0
  return v;
}

// Decoder counterpart.
struct DecodedTrigger {
  uint32_t size;
  uint32_t srcOffset;
  uint32_t dstOffset;
  uint32_t srcMemId;
  uint32_t dstMemId;
  uint32_t type;
  uint32_t tenantId;
  uint32_t semId;
  uint32_t reservedBit;
};

inline DecodedTrigger decode(uint64_t fst, uint64_t snd) {
  constexpr uint64_t maskSize = (1ULL << TriggerBitsSize) - 1;
  constexpr uint64_t maskOffset = (1ULL << TriggerBitsOffset) - 1;
  constexpr uint64_t maskMemId = (1ULL << TriggerBitsMemoryId) - 1;
  constexpr uint64_t maskType = (1ULL << TriggerBitsType) - 1;
  constexpr uint64_t maskTenantId = (1ULL << TriggerBitsTenantId) - 1;
  constexpr uint64_t maskSemId = (1ULL << TriggerBitsSemaphoreId) - 1;

  DecodedTrigger d{};
  d.size = static_cast<uint32_t>(fst & maskSize);
  d.srcOffset = static_cast<uint32_t>((fst >> TriggerBitsSize) & maskOffset);
  d.dstOffset = static_cast<uint32_t>(snd & maskOffset);
  d.srcMemId = static_cast<uint32_t>((snd >> TriggerBitsOffset) & maskMemId);
  d.dstMemId = static_cast<uint32_t>((snd >> (TriggerBitsOffset + TriggerBitsMemoryId)) & maskMemId);
  d.type = static_cast<uint32_t>((snd >> (TriggerBitsOffset + 2 * TriggerBitsMemoryId)) & maskType);
  d.tenantId = static_cast<uint32_t>(
      (snd >> (TriggerBitsOffset + 2 * TriggerBitsMemoryId + TriggerBitsType)) & maskTenantId);
  d.semId = static_cast<uint32_t>(
      (snd >> (TriggerBitsOffset + 2 * TriggerBitsMemoryId + TriggerBitsType + TriggerBitsTenantId)) &
      maskSemId);
  d.reservedBit = static_cast<uint32_t>((snd >> 63) & 0x1);
  return d;
}

}  // namespace

// --- 1) Bit budget sanity: all fields + reserved bit sum to 64 on snd.
TEST(ProxyTriggerLayoutTest, BitBudgetSumsTo64) {
  constexpr unsigned int sndBits = TriggerBitsOffset + 2 * TriggerBitsMemoryId + TriggerBitsType +
                                   TriggerBitsTenantId + TriggerBitsSemaphoreId +
                                   TriggerBitsFifoReserved;
  EXPECT_EQ(sndBits, 64u);
  // And tenant + sem + reserved fills the upper 11 bits exactly.
  EXPECT_EQ(TriggerBitsTenantId + TriggerBitsSemaphoreId + TriggerBitsFifoReserved, 11u);
  // Design v0.2.1 fixes: tenant=4 / sem=6 / reserved=1.
  EXPECT_EQ(TriggerBitsTenantId, 4u);
  EXPECT_EQ(TriggerBitsSemaphoreId, 6u);
  EXPECT_EQ(TriggerBitsFifoReserved, 1u);
}

// --- 2) Round-trip random-ish samples through encode + decode.
TEST(ProxyTriggerLayoutTest, EncodeDecodeRoundTrip) {
  struct Sample {
    uint32_t size, srcOffset, dstOffset, srcMemId, dstMemId, type, tenantId, semId;
  };
  // Boundary values: 0, 1, max, max-1 across each field.
  Sample samples[] = {
      {1, 0, 0, 0, 0, (uint32_t)TriggerData, 0, 0},
      {1024, 64, 128, 1, 2, (uint32_t)TriggerFlag, 1, 1},
      {(1U << 24), 1234, 5678, 7, 13, (uint32_t)TriggerSync, 7, 31},
      // Field maxima.
      {(1U << TriggerBitsSize) - 1, (1U << TriggerBitsOffset) - 1, (1U << TriggerBitsOffset) - 1,
       (1U << TriggerBitsMemoryId) - 1, (1U << TriggerBitsMemoryId) - 1,
       (1U << TriggerBitsType) - 1, (1U << TriggerBitsTenantId) - 1,
       (1U << TriggerBitsSemaphoreId) - 1},
  };

  for (const auto& s : samples) {
    uint64_t fst = encodeFst(s.size, s.srcOffset);
    uint64_t snd = encodeSnd(s.dstOffset, s.srcMemId, s.dstMemId, s.type, s.tenantId, s.semId);

    DecodedTrigger d = decode(fst, snd);
    EXPECT_EQ(d.size, s.size);
    EXPECT_EQ(d.srcOffset, s.srcOffset);
    EXPECT_EQ(d.dstOffset, s.dstOffset);
    EXPECT_EQ(d.srcMemId, s.srcMemId);
    EXPECT_EQ(d.dstMemId, s.dstMemId);
    EXPECT_EQ(d.type, s.type);
    EXPECT_EQ(d.tenantId, s.tenantId);
    EXPECT_EQ(d.semId, s.semId);
    // Critical invariant: business fields must NOT touch the reserved bit.
    EXPECT_EQ(d.reservedBit, 0u);
  }
}

// --- 3) Cross-check explicit shift/mask against the union/bitfield path.
// On every supported compiler the host union ought to produce identical bytes.
// If a future compiler/ABI breaks this, the test fails loudly rather than the
// host proxy silently mis-parsing GPU-pushed triggers.
TEST(ProxyTriggerLayoutTest, BitfieldAgreesWithExplicitEncoder) {
  struct Sample {
    uint32_t size, srcOffset, dstOffset, srcMemId, dstMemId, type, tenantId, semId;
  };
  Sample samples[] = {
      {1, 0, 0, 0, 0, (uint32_t)TriggerData, 0, 0},
      {65536, 64, 128, 1, 2, (uint32_t)TriggerFlag, 1, 1},
      {(1U << 20), 1234, 5678, 7, 13, (uint32_t)TriggerSync, 7, 31},
      {(1U << TriggerBitsSize) - 1, (1U << TriggerBitsOffset) - 1, (1U << TriggerBitsOffset) - 1,
       (1U << TriggerBitsMemoryId) - 1, (1U << TriggerBitsMemoryId) - 1,
       (1U << TriggerBitsType) - 1, (1U << TriggerBitsTenantId) - 1,
       (1U << TriggerBitsSemaphoreId) - 1},
  };

  for (const auto& s : samples) {
    // Populate the union via the C++ bitfield path.
    ProxyTrigger t{};
    t.fields.size = s.size;
    t.fields.srcOffset = s.srcOffset;
    t.fields.dstOffset = s.dstOffset;
    t.fields.srcMemoryId = s.srcMemId;
    t.fields.dstMemoryId = s.dstMemId;
    t.fields.type = s.type;
    t.fields.tenantId = s.tenantId;
    t.fields.semaphoreId = s.semId;

    // Expected raw 64-bit values from the explicit encoder.
    uint64_t expectedFst = encodeFst(s.size, s.srcOffset);
    uint64_t expectedSnd = encodeSnd(s.dstOffset, s.srcMemId, s.dstMemId, s.type, s.tenantId, s.semId);

    EXPECT_EQ(t.fst, expectedFst);
    EXPECT_EQ(t.snd, expectedSnd);

    // And the union view of the bitfield path agrees with our decoder.
    DecodedTrigger d = decode(t.fst, t.snd);
    EXPECT_EQ(d.tenantId, s.tenantId);
    EXPECT_EQ(d.semId, s.semId);
    EXPECT_EQ(d.reservedBit, 0u);
  }
}

// --- 4) Reserved-bit invariant under push-time XOR.
// Simulate what FifoDeviceHandle::push() does (XOR bit 63), then verify the
// host's read-back-and-restore round-trips back to the original business value.
TEST(ProxyTriggerLayoutTest, MsbFlipRoundTripsAndDoesNotPolluteBusinessFields) {
  ProxyTrigger t{};
  t.fields.size = 1024;
  t.fields.tenantId = 9;          // arbitrary in [0, 15]
  t.fields.semaphoreId = 42;      // arbitrary in [0, 63]
  t.fields.type = TriggerSync;
  t.fields.dstOffset = 0xCAFE;
  t.fields.srcMemoryId = 5;
  t.fields.dstMemoryId = 6;

  uint64_t originalSnd = t.snd;
  // Pre-push: reserved bit must be 0.
  EXPECT_EQ((originalSnd >> 63) & 0x1ULL, 0ULL);

  // Simulate push() flipping bit 63.
  uint64_t pushed = originalSnd ^ (1ULL << 63);
  EXPECT_NE(pushed, 0ULL);                                // by construction
  EXPECT_EQ((pushed >> 63) & 0x1ULL, 1ULL);

  // Host service-thread restores by XOR-back.
  uint64_t restored = pushed ^ (1ULL << 63);
  EXPECT_EQ(restored, originalSnd);

  // Decoded fields after restore unchanged.
  DecodedTrigger d = decode(t.fst, restored);
  EXPECT_EQ(d.tenantId, 9u);
  EXPECT_EQ(d.semId, 42u);
  EXPECT_EQ(d.type, (uint32_t)TriggerSync);
  EXPECT_EQ(d.dstOffset, 0xCAFEu);
}
