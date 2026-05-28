// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/errors.hpp>
#include <mscclpp/fifo_device.hpp>
#include <mscclpp/numa.hpp>
#include <mscclpp/port_channel.hpp>

#include "api.h"
#include "atomic.hpp"
#include "connection.hpp"
#include "logger.hpp"
#include "proxy_impl.hpp"

namespace mscclpp {

namespace {
//NOTE: 添加对 semaphores memory tenantId的越界保护
// MT-MSCCL++ (design.md §5.2, v0.2.1): host-side guards mirroring the
// ProxyTrigger bit-field widths in fifo_device.hpp. Hard-throw on overflow
// rather than silent truncation — iter4 lost a day to a silent semId
// wrap that signalled the wrong semaphore and hung the GPU.
constexpr size_t kMaxSemaphoresPerProxy = 1ULL << TriggerBitsSemaphoreId;   // 64
constexpr size_t kMaxMemoriesPerProxy   = 1ULL << TriggerBitsMemoryId;      // 512
constexpr uint32_t kMaxTenantId         = (1U << TriggerBitsTenantId) - 1; // 15
}  // namespace

MSCCLPP_API_CPP void BasePortChannel::setTenantId(uint32_t tenantId) {
  if (tenantId > kMaxTenantId) {
    throw Error("BasePortChannel::setTenantId(" + std::to_string(tenantId) +
                    ") out of range; tenant_id is " + std::to_string(TriggerBitsTenantId) +
                    " bits (max " + std::to_string(kMaxTenantId) + ").",
                ErrorCode::InvalidUsage);
  }
  tenantId_ = tenantId;
}

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(SemaphoreId semaphoreId,
                                                 std::shared_ptr<Host2DeviceSemaphore> semaphore,
                                                 std::shared_ptr<Proxy> proxy)
    : semaphoreId_(semaphoreId), semaphore_(semaphore), proxy_(proxy) {}

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore,
                                                 std::shared_ptr<Proxy> proxy)
    : BasePortChannel(semaphoreId, std::make_shared<Host2DeviceSemaphore>(semaphore), proxy) {}

MSCCLPP_API_CPP PortChannel::PortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                                         std::shared_ptr<Proxy> proxy, MemoryId dst, MemoryId src)
    : BasePortChannel(semaphoreId, semaphore, proxy), dst_(dst), src_(src) {}

MSCCLPP_API_CPP PortChannel::PortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore,
                                         std::shared_ptr<Proxy> proxy, MemoryId dst, MemoryId src)
    : BasePortChannel(semaphoreId, semaphore, proxy), dst_(dst), src_(src) {}

MSCCLPP_API_CPP ProxyService::ProxyService(int fifoSize) {
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  int deviceNumaNode = getDeviceNumaNode(cudaDevice);
  auto initFunc = [cudaDevice, deviceNumaNode]() {
    MSCCLPP_CUDATHROW(cudaSetDevice(cudaDevice));
    if (deviceNumaNode >= 0) {
      numaBind(deviceNumaNode);
      INFO(CONN, "NUMA node of ProxyService proxy thread is set to ", deviceNumaNode);
    }
  };
  //NOTE: PorxySerice 接入 context handler
  //UNKOWN: 为啥 TriggerSync 使用 ctx.fifopos 
  // Legacy path kept for users who plug in a plain ProxyHandler decorator —
  // for them we never see scheduler delay, so picking up the current tail()
  // here is correct.
  auto handlerFunc = [this](ProxyTrigger triggerRaw) {
    ProxyFifoContext ctx{};
    ctx.fifoPos = proxy_->fifo()->tail();
    ctx.enqueueNs = 0;
    return handleTrigger(triggerRaw, ctx);
  };
  //NOTE: 创建了一个proxy对象，proxy对象poll FIFO, 读GPU推来的trigger
  proxy_ = std::make_shared<Proxy>(handlerFunc, initFunc, fifoSize);
  // MT-MSCCL++ (design.md §5.7, v0.2.1): also install a context-aware
  // handler so that the proxy thread can pass the poll-time fifoPos through
  // to handleTrigger without a redundant tail() read. The proxy thread's
  // start() prefers `contextHandler` over `handler` when both are set.
  proxy_->pimpl_->setContextHandler(
      [this](ProxyTrigger triggerRaw, ProxyFifoContext ctx) { return handleTrigger(triggerRaw, ctx); });
  // Combined progress handler: built-in flush progress + optional extra hook
  // installed via setExtraProgressHook() (used by TenantAwareProxyService for
  // token-bucket retry / aging-aware re-pick — design.md §5.3.3 v0.2.1).
  proxy_->pimpl_->setProgressHandler([this]() {
    progressFlushes();
    if (extraProgressHook_) extraProgressHook_();
  });
}

MSCCLPP_API_CPP void ProxyService::setExtraProgressHook(std::function<void()> hook) {
  // Stored on `this`; the proxy thread's progressHandler closure already
  // captures `this` and tests for emptiness on each call (see ctor above).
  extraProgressHook_ = std::move(hook);
}

MSCCLPP_API_CPP SemaphoreId ProxyService::buildAndAddSemaphore(Communicator& communicator,
                                                               const Connection& connection) {
  if (semaphores_.size() >= kMaxSemaphoresPerProxy) {
    throw Error(
        "ProxyService::buildAndAddSemaphore would exceed the per-ProxyService semaphore cap "
        "(" + std::to_string(kMaxSemaphoresPerProxy) + "). Recreate the ProxyService per "
        "benchmark size, or use multiple ProxyService shards. See design.md §5.2 / §10.1.",
        ErrorCode::InvalidUsage);
  }
  semaphores_.push_back(std::make_shared<Host2DeviceSemaphore>(communicator, connection));
  return semaphores_.size() - 1;
}

MSCCLPP_API_CPP SemaphoreId ProxyService::addSemaphore(const Semaphore& semaphore) {
  if (semaphores_.size() >= kMaxSemaphoresPerProxy) {
    throw Error("ProxyService::addSemaphore would exceed the per-ProxyService semaphore cap (" +
                    std::to_string(kMaxSemaphoresPerProxy) + ").",
                ErrorCode::InvalidUsage);
  }
  semaphores_.push_back(std::make_shared<Host2DeviceSemaphore>(semaphore));
  return semaphores_.size() - 1;
}

MSCCLPP_API_CPP SemaphoreId ProxyService::addSemaphore(std::shared_ptr<Host2DeviceSemaphore> semaphore) {
  if (semaphores_.size() >= kMaxSemaphoresPerProxy) {
    throw Error("ProxyService::addSemaphore would exceed the per-ProxyService semaphore cap (" +
                    std::to_string(kMaxSemaphoresPerProxy) + ").",
                ErrorCode::InvalidUsage);
  }
  semaphores_.push_back(semaphore);
  return semaphores_.size() - 1;
}

MSCCLPP_API_CPP MemoryId ProxyService::addMemory(RegisteredMemory memory) {
  if (memories_.size() >= kMaxMemoriesPerProxy) {
    throw Error("ProxyService::addMemory would exceed the per-ProxyService memory cap (" +
                    std::to_string(kMaxMemoriesPerProxy) +
                    "). MemoryId is encoded in " + std::to_string(TriggerBitsMemoryId) + " bits.",
                ErrorCode::InvalidUsage);
  }
  memories_.push_back(memory);
  return memories_.size() - 1;
}

MSCCLPP_API_CPP MemoryId ProxyService::nextMemoryId([[maybe_unused]] uint32_t count) const {
  if (count == 0) {
    throw Error("count must be greater than 0", ErrorCode::InvalidUsage);
  }
  MemoryId firstId = memories_.size();
  return firstId;
}

MSCCLPP_API_CPP std::shared_ptr<Host2DeviceSemaphore> ProxyService::semaphore(SemaphoreId id) const {
  return semaphores_[id];
}

MSCCLPP_API_CPP BasePortChannel ProxyService::basePortChannel(SemaphoreId id) {
  return BasePortChannel(id, semaphores_[id], proxy_);
}

MSCCLPP_API_CPP PortChannel ProxyService::portChannel(SemaphoreId id, MemoryId dst, MemoryId src) {
  return PortChannel(id, semaphores_[id], proxy_, dst, src);
}

MSCCLPP_API_CPP void ProxyService::setHandlerDecorator(ProxyHandlerDecorator decorator) {
  if (!decorator) return;
  // Legacy decorator path: the wrapped inner sees only the trigger, and
  // reads tail() itself for fifoPos. This is OK only when the decorator
  // does NOT delay dispatch — otherwise see setContextHandlerDecorator().
  ProxyHandler inner = [this](ProxyTrigger triggerRaw) {
    ProxyFifoContext ctx{};
    ctx.fifoPos = proxy_->fifo()->tail();
    ctx.enqueueNs = 0;
    return handleTrigger(triggerRaw, ctx);
  };
  proxy_->pimpl_->handler = decorator(std::move(inner));
  // Disable the context handler — legacy decorator wins; the proxy thread's
  // start() loop will fall through to `handler` because contextHandler is
  // cleared.
  proxy_->pimpl_->contextHandler = nullptr;
}

MSCCLPP_API_CPP void ProxyService::setContextHandlerDecorator(ContextProxyHandlerDecorator decorator) {
  if (!decorator) return;
  // Context-aware decorator path (design.md §5.7, v0.2.1). The inner takes
  // both the trigger and its poll-time context, and the decorator may store
  // the context (e.g. inside a PendingTrigger) and replay it later when it
  // actually dispatches the trigger. handleTrigger receives `ctx` and uses
  // ctx.fifoPos for TriggerSync flush boundaries instead of fifo->tail().
  ContextProxyHandler inner = [this](ProxyTrigger triggerRaw, ProxyFifoContext ctx) {
    return handleTrigger(triggerRaw, ctx);
  };
  proxy_->pimpl_->contextHandler = decorator(std::move(inner));
}

MSCCLPP_API_CPP void ProxyService::startProxy(bool blocking) { proxy_->start(blocking); }

MSCCLPP_API_CPP void ProxyService::stopProxy() {
  proxy_->stop();
  // Drain pending flushes. After a bounded loop, force-unblock any still-pending GPU
  // waiters with a sentinel write (UINT64_MAX > any FIFO position).
  for (int i = 0; i < 1000 && !pendingFlushPos_.empty(); ++i) {
    progressFlushes();
  }
  if (!pendingFlushPos_.empty()) {
    WARN(CONN, "stopProxy: ", pendingFlushPos_.size(), " connections still pending; writing sentinel");
    for (auto& [conn, pos] : pendingFlushPos_) {
      if (uint64_t* ptr = conn->getFlushDonePtr()) atomicStore(ptr, UINT64_MAX, memoryOrderRelease);
    }
    pendingFlushPos_.clear();
  }
}

ProxyHandlerResult ProxyService::handleTrigger(ProxyTrigger trigger, ProxyFifoContext ctx) {
  // MT-MSCCL++ (design.md §5.7, v0.2.1): use the caller-provided fifoPos.
  // For the non-tenant path the proxy thread captured this at poll time
  // (== tail() pre-pop). For the tenant-scheduled path, the scheduler
  // captured it at enqueue time and is replaying it now, possibly long
  // after fifo->tail() has advanced past this trigger's position.
  // Falling back to tail() here would silently desync the flush boundary
  // by exactly the scheduler's delay.
  uint64_t pos = ctx.fifoPos;

  std::shared_ptr<Host2DeviceSemaphore> semaphore = semaphores_[trigger.fields.semaphoreId];

  auto& conn = semaphore->connection();
  int maxWriteQueueSize = conn.getMaxWriteQueueSize();
  auto& numRequests = inflightRequests_[conn.impl_];

  if (trigger.fields.type & TriggerData) {
    RegisteredMemory& dst = memories_[trigger.fields.dstMemoryId];
    RegisteredMemory& src = memories_[trigger.fields.srcMemoryId];
    conn.write(dst, trigger.fields.dstOffset, src, trigger.fields.srcOffset, trigger.fields.size);
    numRequests++;
  }

  if (trigger.fields.type & TriggerFlag) {
    semaphore->signal();
    numRequests++;
  }

  if (trigger.fields.type & TriggerSync) {
    // Record this TriggerSync's FIFO position. The GPU caller is spinning on
    // flushDonePos_ > pos; progressFlushes() will publish pos+1 once the CQ drains.
    // Later TriggerSyncs on the same conn overwrite — CQ drain completes them all at once.
    conn.impl_->requestFlush();
    pendingFlushPos_[conn.impl_] = pos;
    numRequests = 0;
  } else if (maxWriteQueueSize != -1 && numRequests >= maxWriteQueueSize) {
    conn.flush();  // flow-control flush stays blocking
    numRequests = 0;
  }

  return ProxyHandlerResult::Continue;
}

MSCCLPP_API_CPP BasePortChannel::DeviceHandle BasePortChannel::deviceHandle() const {
  auto& conn = semaphore_->connection();
  return BasePortChannel::DeviceHandle(semaphoreId_, semaphore_->deviceHandle(), proxy_->fifo()->deviceHandle(),
                                       conn.impl_->getFlushDonePtr(), tenantId_);
}

MSCCLPP_API_CPP PortChannel::DeviceHandle PortChannel::deviceHandle() const {
  auto& conn = semaphore_->connection();
  return PortChannel::DeviceHandle(semaphoreId_, semaphore_->deviceHandle(), proxy_->fifo()->deviceHandle(), dst_, src_,
                                   conn.impl_->getFlushDonePtr(), tenantId_);
}

void ProxyService::progressFlushes() {
  for (auto it = pendingFlushPos_.begin(); it != pendingFlushPos_.end();) {
    if (it->first->progressFlush()) {
      // CQ drained: publish pos+1 to unblock GPU waiters whose own pos <= recorded pos.
      atomicStore(it->first->getFlushDonePtr(), it->second + 1, memoryOrderRelease);
      it = pendingFlushPos_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace mscclpp
