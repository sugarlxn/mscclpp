// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// nanobind bindings for MT-MSCCL++ tenant API (design.md §3.5, §5, §6).

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <algorithm>

#include <mscclpp/ext/tenant.hpp>
#include <mscclpp/ext/tenant_aware_proxy.hpp>

namespace nb = nanobind;
using namespace mscclpp;
using namespace mscclpp::ext::tenant;

// 这是一层 nanobind Python binding。它不实现调度逻辑，只负责把 C++ 里的 MT-MSCCL++ tenant API 暴露给 Python
//NOTE: 暴露 QoSClass、PolicyMode、TenantContext、TenantAwareProxyService
void register_tenant(nb::module_& m) {
  nb::enum_<QoSClass>(m, "CppQoSClass")
      .value("BestEffort", QoSClass::BestEffort)
      .value("Standard", QoSClass::Standard)
      .value("Premium", QoSClass::Premium)
      .value("Realtime", QoSClass::Realtime);

  nb::enum_<PolicyMode>(m, "CppPolicyMode")
      .value("SinglePassthrough", PolicyMode::SinglePassthrough)
      .value("Fair", PolicyMode::Fair)
      .value("StrictPriority", PolicyMode::StrictPriority)
      .value("Hybrid", PolicyMode::Hybrid)
      .value("Fifo", PolicyMode::Fifo);

  nb::class_<TenantContext>(m, "CppTenantContext")
      .def(nb::init<>())
      .def_rw("tenant_id", &TenantContext::tenant_id)
      .def_rw("qos_class", &TenantContext::qos_class)
      .def_rw("weight", &TenantContext::weight)
      .def_rw("sla_latency_ns", &TenantContext::sla_latency_ns)
      .def_rw("bandwidth_min_bps", &TenantContext::bandwidth_min_bps)
      .def_rw("bandwidth_max_bps", &TenantContext::bandwidth_max_bps)
      .def_rw("create_ts_ns", &TenantContext::create_ts_ns);

  nb::class_<BandwidthBudget>(m, "CppBandwidthBudget")
      .def(nb::init<>())
      .def_rw("bytes_per_second", &BandwidthBudget::bytes_per_second)
      .def_rw("burst_bytes", &BandwidthBudget::burst_bytes)
      .def_rw("last_refill_ts_ns", &BandwidthBudget::last_refill_ts_ns);

  // TenantAwareProxyService derives from ProxyService, so the BaseProxyService
  // parent allows polymorphism with the existing CppBaseProxyService binding.
  nb::class_<TenantAwareProxyService, ProxyService>(m, "CppTenantAwareProxyService")
      .def(nb::init<PolicyMode, int, uint32_t, bool, uint64_t, uint64_t>(),
           nb::arg("mode") = PolicyMode::SinglePassthrough,
           nb::arg("fifo_size") = DEFAULT_FIFO_SIZE,
           nb::arg("scheduling_window_size") = DEFAULT_SCHEDULING_WINDOW_SIZE, nb::arg("debug") = false,
           nb::arg("small_collective_threshold_bytes") = DEFAULT_SMALL_COLLECTIVE_THRESHOLD_BYTES,
           nb::arg("aging_ns") = DEFAULT_AGING_NS)
      .def("update_tenant", &TenantAwareProxyService::updateTenant, nb::arg("ctx"), nb::arg("budget"))
      .def("register_tenant", &TenantAwareProxyService::registerTenant, nb::arg("tenant_id"), nb::arg("qos"),
           nb::arg("weight") = 1, nb::arg("bandwidth_max_bps") = uint64_t{0}, nb::arg("burst_bytes") = uint64_t{0})
      .def("remove_tenant", &TenantAwareProxyService::removeTenant, nb::arg("tenant_id"))
      .def("set_mode", &TenantAwareProxyService::setMode, nb::arg("mode"))
      .def("set_tenant_collective_bytes", &TenantAwareProxyService::setTenantCollectiveBytes,
           nb::arg("tenant_id"), nb::arg("bytes"))
      .def("set_small_collective_threshold_bytes", &TenantAwareProxyService::setSmallCollectiveThresholdBytes,
           nb::arg("bytes"))
      .def("small_collective_threshold_bytes", &TenantAwareProxyService::smallCollectiveThresholdBytes)
      .def("set_debug", &TenantAwareProxyService::setDebug, nb::arg("enabled"))
      .def("debug_enabled", &TenantAwareProxyService::debugEnabled)
      .def("scheduler_debug_counters",
           [](const TenantAwareProxyService& svc) {
             auto counters = svc.schedulerDebugCounters();
             nb::dict out;
             for (uint32_t tid = 0; tid < MAX_TENANTS; ++tid) {
               const auto& c = counters[tid];
               nb::dict row;
               row["size_aware_bypass_triggers"] = c.size_aware_bypass_triggers;
               row["size_aware_bypass_bytes"] = c.size_aware_bypass_bytes;
               row["sched_dispatched_triggers"] = c.sched_dispatched_triggers;
               row["sched_dispatched_bytes"] = c.sched_dispatched_bytes;
               row["token_bucket_waits"] = c.token_bucket_waits;
               row["drr_picks"] = c.drr_picks;
               row["strict_priority_picks"] = c.strict_priority_picks;
               row["scheduler_wait_samples"] = c.scheduler_wait_samples;
               row["scheduler_wait_avg_ns"] =
                   c.scheduler_wait_samples ? c.scheduler_wait_total_ns / c.scheduler_wait_samples : uint64_t{0};
               auto samples = c.scheduler_wait_ns_samples;
               if (!samples.empty()) {
                 std::sort(samples.begin(), samples.end());
                 auto pct = [&samples](double q) -> uint64_t {
                   size_t idx = static_cast<size_t>(q * static_cast<double>(samples.size() - 1));
                   return samples[idx];
                 };
                 row["scheduler_wait_p50_ns"] = pct(0.50);
                 row["scheduler_wait_p99_ns"] = pct(0.99);
               } else {
                 row["scheduler_wait_p50_ns"] = uint64_t{0};
                 row["scheduler_wait_p99_ns"] = uint64_t{0};
               }
               out[nb::int_(tid)] = row;
             }
             return out;
           })
      .def("mode", &TenantAwareProxyService::mode);
}
