import time
import torch
import warnings
class CudaTimer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self._cpu_t0 = None
        self._cpu_t1 = None

        self._start_event = None
        self._end_event = None
        if self.enabled:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)

    def cuda_start(self):
        if self.enabled:
            # record timestamp on the current CUDA stream
            self._start_event.record()
        else:
            warnings.warn("CUDA not available, CudaTimer will use CPU time. Timer is not active")
            
    def cuda_stop(self):
        if self.enabled:
            self._end_event.record()
        else:
            warnings.warn("CUDA not available, CudaTimer will use CPU time. Timer is not active")

    def cpu_start(self):
        self._cpu_t0 = time.perf_counter()
        
    
    def cpu_stop(self):
        self._cpu_t1 = time.perf_counter()
        

    def cpu_elapsed_ms(self) -> float:
        if self._cpu_t0 is None or self._cpu_t1 is None:
            raise ValueError("CPU timer was not properly started and stopped.")
        return (self._cpu_t1 - self._cpu_t0) * 1000.0
    
    def cuda_elapsed_ms(self) -> float:
        if self.enabled:
            # only sync when you actually need the number
            self._end_event.synchronize()
            return self._start_event.elapsed_time(self._end_event)  # milliseconds
        else:
            warnings.warn("CUDA not available, CudaTimer will use CPU time. Timer is not active")

    # -- Unified interface (CUDA if available, CPU fallback) --

    def start(self) -> float:
        """Unified start: CUDA events if available, CPU perf_counter otherwise."""
        self.cpu_start()
        if self.enabled:
            self.cuda_start()
        return self._cpu_t0

    def stop(self):
        if self.enabled:
            self.cuda_stop()
        self.cpu_stop()

    def elapsed_ms(self) -> float:
        if self.enabled:
            return self.cuda_elapsed_ms()
        return self.cpu_elapsed_ms()
