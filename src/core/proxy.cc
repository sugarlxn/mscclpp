// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <chrono>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/numa.hpp>
#include <mscclpp/proxy.hpp>
#include <mscclpp/utils.hpp>

#include "api.h"
#include "logger.hpp"
#include "proxy_impl.hpp"

namespace mscclpp {

constexpr int ProxyStopCheckPeriod = 1000;
constexpr int ProxyStartWarnPeriod = 1000;

MSCCLPP_API_CPP Proxy::Proxy(ProxyHandler handler, std::function<void()> threadInit, int fifoSize) {
  pimpl_ = std::make_unique<Impl>(handler, threadInit, fifoSize);
}

MSCCLPP_API_CPP Proxy::Proxy(ProxyHandler handler, int fifoSize) {
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  int deviceNumaNode = getDeviceNumaNode(cudaDevice);
  auto initFunc = [cudaDevice, deviceNumaNode]() {
    MSCCLPP_CUDATHROW(cudaSetDevice(cudaDevice));
    if (deviceNumaNode >= 0) {
      numaBind(deviceNumaNode);
    }
  };
  pimpl_ = std::make_unique<Impl>(handler, initFunc, fifoSize);
}

MSCCLPP_API_CPP Proxy::~Proxy() {
  if (pimpl_) {
    stop();
  }
}

MSCCLPP_API_CPP void Proxy::start(bool blocking) {
  pimpl_->running.store(true, std::memory_order_release);
  pimpl_->service = std::thread([this] {
    // threadInit() is responsible for setting up the runtime context for the thread.
    // The default implementation sets the CUDA device and NUMA affinity to match the main thread (see Proxy ctor).
    // It should be called before any CUDA API calls to avoid resource allocation on unwanted GPUs.
    pimpl_->threadInit();

    // never capture in a proxy thread
    auto mode = cudaStreamCaptureModeRelaxed;
    MSCCLPP_CUDATHROW(cudaThreadExchangeStreamCaptureMode(&mode));

    pimpl_->threadStarted.store(true, std::memory_order_release);

    // Snapshot handler pointers at thread start (avoid per-iteration atomics).
    // ctxHandler takes priority when set; otherwise legacy handler is used.
    ProxyHandler handler = pimpl_->handler;
    ContextProxyHandler ctxHandler = pimpl_->contextHandler;
    auto progressHandler = pimpl_->progressHandler;
    auto fifo = pimpl_->fifo;
    ProxyTrigger trigger;

    int runCnt = ProxyStopCheckPeriod;
    for (;;) {
      if (runCnt-- == 0) {
        runCnt = ProxyStopCheckPeriod;
        if (!pimpl_->running.load(std::memory_order_acquire)) {
          break;
        }
      }
      // Per-iteration system work (e.g. progressing pending async flushes),
      // run regardless of whether a trigger is ready this iteration.
      if (progressHandler) progressHandler();

      // Poll to see if we are ready to send anything
      trigger = fifo->poll();
      if (trigger.fst == 0 || trigger.snd == 0) {  // TODO: this check is a potential pitfall for custom triggers
        continue;                                  // there is one in progress
      }
      trigger.snd ^= (uint64_t{1} << uint64_t{63});  // this is where the last bit of snd is reverted.

      // MT-MSCCL++ (design.md §5.7, v0.2.1): capture fifoPos BEFORE the
      // handler may delay dispatch (e.g. via TenantAwareProxyHandler). At
      // this point the proxy thread is the sole FIFO consumer and has not
      // yet called pop(), so `fifo->tail()` is the exact push-time position
      // of this trigger. After pop() runs (below), `tail()` advances and is
      // no longer trustworthy for delayed dispatchers.
      ProxyFifoContext ctx{};
      ctx.fifoPos = fifo->tail();
      ctx.enqueueNs = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count());

      ProxyHandlerResult result = ctxHandler ? ctxHandler(trigger, ctx) : handler(trigger);

      // Send completion: reset only the high 64 bits
      fifo->pop();

      if (result == ProxyHandlerResult::Stop) {
        break;
      }
    }
  });

  if (blocking) {
    int count = ProxyStartWarnPeriod;
    while (!pimpl_->threadStarted.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      count--;
      if (count == 0) {
        count = ProxyStartWarnPeriod;
        WARN(CONN, "Proxy thread startup taking longer than expected.");
      }
    }
  }
}

MSCCLPP_API_CPP void Proxy::stop() {
  pimpl_->running.store(false, std::memory_order_release);
  if (pimpl_->service.joinable()) {
    pimpl_->service.join();
  }
  pimpl_->threadStarted.store(false, std::memory_order_release);
}

MSCCLPP_API_CPP std::shared_ptr<Fifo> Proxy::fifo() { return pimpl_->fifo; }

}  // namespace mscclpp
