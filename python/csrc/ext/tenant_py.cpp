// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// nanobind bindings for MT-MSCCL++ tenant API (design.md §3.5, §5, §6).

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mscclpp/ext/tenant.hpp>
#include <mscclpp/ext/tenant_aware_proxy.hpp>

namespace nb = nanobind;
using namespace mscclpp;
using namespace mscclpp::ext::tenant;

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
      .value("Hybrid", PolicyMode::Hybrid);

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
      .def(nb::init<PolicyMode, int>(), nb::arg("mode") = PolicyMode::SinglePassthrough,
           nb::arg("fifo_size") = DEFAULT_FIFO_SIZE)
      .def("update_tenant", &TenantAwareProxyService::updateTenant, nb::arg("ctx"), nb::arg("budget"))
      .def("register_tenant", &TenantAwareProxyService::registerTenant, nb::arg("tenant_id"), nb::arg("qos"),
           nb::arg("weight") = 1, nb::arg("bandwidth_max_bps") = uint64_t{0}, nb::arg("burst_bytes") = uint64_t{0})
      .def("remove_tenant", &TenantAwareProxyService::removeTenant, nb::arg("tenant_id"))
      .def("set_mode", &TenantAwareProxyService::setMode, nb::arg("mode"))
      .def("mode", &TenantAwareProxyService::mode);
}
