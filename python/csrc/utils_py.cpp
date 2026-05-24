// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/utils.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_utils(nb::module_& m) {
  m.def("get_host_name", &getHostName, nb::arg("maxlen"), nb::arg("delim"));
  // MT-MSCCL++ (iter 6 fix): expose the exact byte size of DeviceSyncer so
  // Python wrappers (MscclppAllReduce3) can allocate per-instance buffers
  // without hardcoding a literal. The C++ side has a static_assert that
  // matches; Python uses this to size cp.zeros(...). If the struct ever
  // grows, both ends fail loudly at compile / first-load time.
  m.attr("DEVICE_SYNCER_SIZE_BYTES") = static_cast<int>(DeviceSyncerSizeBytes);
}
