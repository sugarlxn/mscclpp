// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// MT-MSCCL++: multi-tenant extensions to MSCCL++.
// See doc/design.md §3.5, §5 for the full design.
//
// Iteration 1: declarations only. The reference scheduler/rate-limiter live in
// python/mscclpp/ext/tenant.py and orchestrate at the host/Python level so the
// MSCCL++ device-side ABI stays untouched. Iteration 2 will lower
// TenantScheduler into ProxyService (see design.md §7.1 / §8 Phase 2).

#ifndef MSCCLPP_EXT_TENANT_HPP_
#define MSCCLPP_EXT_TENANT_HPP_

#include <cstdint>

namespace mscclpp {
namespace ext {
namespace tenant {

//NOTE: 新增 tenant 数据模型：TenantId、QoSClass、PolicyMode、TenantContext、BandwidthBudget、PolicyTable
using TenantId = uint8_t;

constexpr TenantId DEFAULT_TENANT = 0;
constexpr unsigned int MAX_TENANTS = 16;  // 4-bit tenant_id (design.md §5.2, v0.2.1)

enum class QoSClass : uint8_t {
  BestEffort = 0,  
  Standard   = 1,  
  Premium    = 2,
  Realtime   = 3,
};

enum class PolicyMode : uint8_t {
  SinglePassthrough = 0,  //仅1个活跃租户 + scheduelr 队列为空时 bypass 
  Fair              = 1,  //DRR / DRF 公平调度
  StrictPriority    = 2,  //绝对优先级，带老化机制的 chunk 边界抢占
  Hybrid            = 3,  //先 StrictPriority（仅 Premium/Realtime）优先，再 Fair
};

struct TenantContext {
  TenantId tenant_id;
  QoSClass qos_class;
  uint32_t weight;
  uint64_t sla_latency_ns;
  uint64_t bandwidth_min_bps;
  uint64_t bandwidth_max_bps;
  uint64_t create_ts_ns;
};

struct BandwidthBudget {
  uint64_t bytes_per_second;
  uint64_t burst_bytes;
  uint64_t last_refill_ts_ns;
};

struct PolicyTable {
  PolicyMode mode;
  uint16_t   registered_tenant_mask;
  uint16_t   active_tenant_mask;
  TenantContext   tenants[MAX_TENANTS];
  BandwidthBudget budgets[MAX_TENANTS];
  uint64_t  version;
};

}  // namespace tenant
}  // namespace ext
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_TENANT_HPP_
