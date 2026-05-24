// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PROXY_HPP_
#define MSCCLPP_PROXY_HPP_

#include <cstdint>
#include <functional>
#include <memory>

#include "fifo.hpp"

namespace mscclpp {

/// Return values for ProxyHandler.
enum class ProxyHandlerResult {
  /// Move to next trigger in FIFO.
  Continue,
  /// Stop and exit proxy.
  Stop,
};

class Proxy;
class ProxyService;

/// Handler function type for proxy. Called once per ready FIFO trigger.
using ProxyHandler = std::function<ProxyHandlerResult(ProxyTrigger)>;

/// MT-MSCCL++ (design.md §5.7, v0.2.1): metadata captured AT POLL TIME — i.e.
/// before the proxy thread calls `pop()` and before any tenant scheduler may
/// delay dispatch. This is the only reliable place to record the FIFO
/// position that corresponds 1:1 with the device-side `prevHead` returned by
/// `FifoDeviceHandle::push()`. After scheduler-induced delay, `fifo->tail()`
/// is the position of some *later* trigger, so handlers that need true
/// per-trigger ordering (TriggerSync flush boundary; per-connection FIFO)
/// must take fifoPos from this struct, not from a fresh tail() read.
struct ProxyFifoContext {
  /// Monotonic FIFO position of this trigger as assigned by the GPU-side
  /// push(). Equal to `fifo->tail()` at the moment the proxy thread polled
  /// this trigger.
  uint64_t fifoPos;
  /// Host monotonic timestamp (ns) when the proxy thread observed this
  /// trigger. Used by `TenantAwareProxyHandler` for aging / progress-hook
  /// retries.
  uint64_t enqueueNs;
};

/// Context-aware handler. Called once per ready FIFO trigger, passing both
/// the trigger itself and its per-poll context. Preferred over `ProxyHandler`
/// when downstream logic (e.g. tenant scheduling) may delay dispatch and
/// therefore needs the original FIFO position.
using ContextProxyHandler = std::function<ProxyHandlerResult(ProxyTrigger, ProxyFifoContext)>;

/// Host-side proxy for PortChannels.
class Proxy {
 public:
  /// Constructor.
  /// @param handler Handler for each FIFO trigger.
  /// @param threadInit Optional function run once in the proxy thread before FIFO consumption.
  ///        The function should initialize thread runtime context before any CUDA API call in that thread
  ///        (for example, set CUDA device and optionally bind NUMA affinity).
  /// @param fifoSize FIFO size (default: DEFAULT_FIFO_SIZE).
  Proxy(ProxyHandler handler, std::function<void()> threadInit, int fifoSize = DEFAULT_FIFO_SIZE);

  /// Constructor.
  /// @param handler Handler for each FIFO trigger.
  /// @param fifoSize FIFO size (default: DEFAULT_FIFO_SIZE).
  Proxy(ProxyHandler handler, int fifoSize = DEFAULT_FIFO_SIZE);

  /// Destructor. Stops proxy if running.
  ~Proxy();

  /// Start proxy.
  void start(bool blocking = false);

  /// Stop proxy.
  void stop();

  /// Get reference to FIFO used by proxy.
  /// @return Shared pointer to FIFO.
  std::shared_ptr<Fifo> fifo();

 private:
  friend class ProxyService;
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_PROXY_HPP_
