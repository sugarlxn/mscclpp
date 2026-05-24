// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PORT_CHANNEL_HPP_
#define MSCCLPP_PORT_CHANNEL_HPP_

#include "core.hpp"
#include "port_channel_device.hpp"
#include "proxy.hpp"
#include "semaphore.hpp"

namespace mscclpp {

struct BasePortChannel;
struct PortChannel;

/// MT-MSCCL++: a decorator that wraps the ProxyService's internal trigger
/// handler with custom logic (e.g. tenant-aware scheduling). The decorator
/// receives the inner handler and returns a new outer handler. See
/// include/mscclpp/ext/tenant_aware_proxy.hpp for the canonical use.
using ProxyHandlerDecorator = std::function<ProxyHandler(ProxyHandler)>;

/// MT-MSCCL++ (design.md §5.7, v0.2.1): the context-aware variant. Use this
/// when the decorator may delay dispatch — it receives a `ContextProxyHandler`
/// (inner) that takes both ProxyTrigger and the per-poll context (fifoPos +
/// enqueueNs), so the inner handler can use the original push-time fifoPos
/// even after scheduler-induced delay.
using ContextProxyHandlerDecorator = std::function<ContextProxyHandler(ContextProxyHandler)>;

/// Base class for proxy services. Proxy services are used to proxy data between devices.
class BaseProxyService {
 public:
  BaseProxyService() = default;
  virtual ~BaseProxyService() = default;
  virtual void startProxy(bool blocking = false) = 0;
  virtual void stopProxy() = 0;
};

/// Proxy service implementation.
class ProxyService : public BaseProxyService {
 public:
  /// Constructor.
  /// @param fifoSize Size of the FIFO used by the proxy service (default: DEFAULT_FIFO_SIZE).
  ProxyService(int fifoSize = DEFAULT_FIFO_SIZE);

  /// Build and add a semaphore to the proxy service.
  /// @param connection The connection associated with the semaphore.
  /// @return The ID of the semaphore.
  SemaphoreId buildAndAddSemaphore(Communicator& communicator, const Connection& connection);

  /// Add a semaphore to the proxy service.
  /// @param semaphore The semaphore to be added
  /// @return The ID of the semaphore.
  SemaphoreId addSemaphore(const Semaphore& semaphore);

  /// Add a semaphore to the proxy service.
  /// @param semaphore The semaphore to be added
  /// @return The ID of the semaphore.
  SemaphoreId addSemaphore(std::shared_ptr<Host2DeviceSemaphore> semaphore);

  /// Register a memory region with the proxy service.
  /// @param memory The memory region to register.
  /// @return The ID of the memory region.
  MemoryId addMemory(RegisteredMemory memory);

  /// Get the next available memory ID.
  /// @param count The number of consecutive IDs required (default: 1).
  /// @return The first ID of an available range [first, first + count).
  MemoryId nextMemoryId(uint32_t count = 1) const;

  /// Get a semaphore by ID.
  /// @param id The ID of the semaphore.
  /// @return The semaphore.
  std::shared_ptr<Host2DeviceSemaphore> semaphore(SemaphoreId id) const;

  /// Get a base port channel by semaphore ID.
  /// @param id The ID of the semaphore.
  /// @return The base port channel.
  BasePortChannel basePortChannel(SemaphoreId id);

  /// Get a port channel by semaphore ID and memory regions.
  /// @param id The ID of the semaphore.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @return The port channel.
  PortChannel portChannel(SemaphoreId id, MemoryId dst, MemoryId src);

  /// Install a decorator that wraps the proxy's trigger handler. MUST be
  /// called BEFORE startProxy(). The decorator is invoked once with the
  /// ProxyService's own handler as `inner`; the returned callable replaces
  /// the proxy thread's trigger handler. Used by MT-MSCCL++ to inject
  /// tenant-aware scheduling. design.md §5.3 / §7.1.
  ///
  /// Legacy form (no fifoPos visibility). Prefer
  /// `setContextHandlerDecorator()` for decorators that may delay dispatch
  /// — those need the per-poll context to keep TriggerSync flush boundaries
  /// correct under delayed scheduling.
  void setHandlerDecorator(ProxyHandlerDecorator decorator);

  /// MT-MSCCL++ (design.md §5.7, v0.2.1): context-aware decorator install.
  /// The `inner` passed to the decorator IS the ProxyService's own
  /// `(trigger, context) -> result` handler that uses `context.fifoPos`
  /// directly (NOT a fresh tail() read), so any delay introduced by the
  /// decorator does not perturb flush boundaries or per-connection FIFO.
  void setContextHandlerDecorator(ContextProxyHandlerDecorator decorator);

  /// MT-MSCCL++ (design.md §5.3.3, v0.2.1): register an extra hook that
  /// the proxy thread calls every poll-loop iteration AFTER the built-in
  /// progressFlushes(). Used by TenantAwareProxyService to retry drain on
  /// rate-limited / aged tenants when no fresh trigger has arrived.
  /// May be called multiple times — only the most recent hook is kept.
  void setExtraProgressHook(std::function<void()> hook);

  /// Start the proxy service.
  /// @param blocking Whether to block until the proxy thread has started (default: false).
  void startProxy(bool blocking = false);

  /// Stop the proxy service.
  void stopProxy();

 protected:
  // Exposed to subclasses (e.g. TenantAwareProxyService) for handler-rewrap.
  std::shared_ptr<Proxy>& proxy() { return proxy_; }
  ProxyHandlerResult dispatchTrigger(ProxyTrigger t) { return handleTrigger(t, ProxyFifoContext{0, 0}); }
  ProxyHandlerResult dispatchTrigger(ProxyTrigger t, ProxyFifoContext ctx) { return handleTrigger(t, ctx); }
  // Exposed so TenantAwareProxyService can chain its own progress tick
  // (token-bucket retry, aging) AFTER the base flush-progress work.
  void runProgressFlushes() { progressFlushes(); }

 private:
  std::vector<std::shared_ptr<Host2DeviceSemaphore>> semaphores_;
  std::vector<RegisteredMemory> memories_;
  std::shared_ptr<Proxy> proxy_;
  std::unordered_map<std::shared_ptr<BaseConnection>, int> inflightRequests_;
  // Latest pending TriggerSync FIFO position per connection. Proxy publishes pos+1 to the
  // connection's gpuFlushDonePos_ when the CQ drains, then erases the entry.
  std::unordered_map<std::shared_ptr<BaseConnection>, uint64_t> pendingFlushPos_;

  // MT-MSCCL++ (design.md §5.7, v0.2.1): handleTrigger now takes the per-poll
  // context. For non-tenant paths the proxy thread passes the fresh
  // poll-time fifoPos; for tenant-aware paths the scheduler passes the
  // PUSH-time fifoPos it captured at enqueue, so TriggerSync's flush
  // boundary stays correct even after dispatch delay.
  ProxyHandlerResult handleTrigger(ProxyTrigger triggerRaw, ProxyFifoContext ctx);
  void progressFlushes();

  // MT-MSCCL++ (v0.2.1): extra progress hook chained after progressFlushes.
  // Set via setExtraProgressHook(); the proxy thread's progressHandler
  // calls progressFlushes() then this hook if non-empty.
  std::function<void()> extraProgressHook_;
};

/// Port channel without specifying source/destination memory regions.
struct BasePortChannel {
 protected:
  SemaphoreId semaphoreId_;

  std::shared_ptr<Host2DeviceSemaphore> semaphore_;

  std::shared_ptr<Proxy> proxy_;

  // MT-MSCCL++ (design.md §5.2): tenant_id propagated into the device handle
  // and ultimately into every ProxyTrigger pushed from this channel. The
  // host-side default of 0 maps to mscclpp::ext::tenant::DEFAULT_TENANT.
  uint32_t tenantId_ = 0;

 public:
  /// Constructor.
  BasePortChannel() = default;

  /// Constructor.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param proxy The proxy used for communication.
  BasePortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                  std::shared_ptr<Proxy> proxy);

  /// Constructor.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param proxy The proxy used for communication.
  BasePortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore, std::shared_ptr<Proxy> proxy);

  /// Copy constructor.
  /// @param other The other BasePortChannel to copy from.
  BasePortChannel(const BasePortChannel& other) = default;

  /// Assignment operator.
  /// @param other The other BasePortChannel to assign from.
  BasePortChannel& operator=(BasePortChannel& other) = default;

  /// MT-MSCCL++: set the tenant_id carried by every trigger this channel
  /// pushes. Must be called BEFORE deviceHandle() is captured by the caller
  /// (the GPU-side handle is a value snapshot). Throws InvalidUsage if
  /// tenantId is out of range; tenantId is encoded in TriggerBitsTenantId
  /// bits (= 4 bits / range [0..15] in design v0.2.1).
  void setTenantId(uint32_t tenantId);
  uint32_t tenantId() const { return tenantId_; }

  /// Device-side handle for BasePortChannel.
  using DeviceHandle = BasePortChannelDeviceHandle;

  /// Returns the device-side handle.
  /// User should make sure the BasePortChannel is not released when using the returned handle.
  /// @return The device-side handle.
  DeviceHandle deviceHandle() const;
};

/// Port channel.
struct PortChannel : public BasePortChannel {
 private:
  MemoryId dst_;
  MemoryId src_;

 public:
  /// Constructor.
  PortChannel() = default;

  /// Constructor.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore.
  /// @param proxy The proxy.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  PortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore, std::shared_ptr<Proxy> proxy,
              MemoryId dst, MemoryId src);

  /// Constructor.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore.
  /// @param proxy The proxy.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  PortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore, std::shared_ptr<Proxy> proxy, MemoryId dst,
              MemoryId src);

  /// Copy constructor.
  /// @param other The other PortChannel to copy from.
  PortChannel(const PortChannel& other) = default;

  /// Assignment operator.
  /// @param other The other PortChannel to assign from.
  PortChannel& operator=(PortChannel& other) = default;

  /// Device-side handle for PortChannel.
  using DeviceHandle = PortChannelDeviceHandle;

  /// Returns the device-side handle.
  /// User should make sure the PortChannel is not released when using the returned handle.
  /// @return The device-side handle.
  DeviceHandle deviceHandle() const;
};

/// @deprecated Use BasePortChannel instead.
[[deprecated("Use BasePortChannel instead.")]] typedef BasePortChannel BaseProxyChannel;

/// @deprecated Use PortChannel instead.
[[deprecated("Use PortChannel instead.")]] typedef PortChannel ProxyChannel;

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_HPP_
