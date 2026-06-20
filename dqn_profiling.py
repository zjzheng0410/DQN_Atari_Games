import json
import threading
import time
from contextlib import nullcontext

import torch


DEFAULT_CUDA_STAGES = {
    "actor_inference",
    "rb.add",
    "rb.sample_many",
    "learner.forward_target",
    "learner.forward_q",
    "learner.backward",
    "optimizer.step",
    "learner.update_total",
}


class _StageTimer:
    def __init__(self, profiler, name, use_cuda):
        self.profiler = profiler
        self.name = name
        self.use_cuda = use_cuda
        self.cpu_start = None
        self.cuda_start = None
        self.cuda_end = None

    def __enter__(self):
        if self.use_cuda and self.profiler.cuda_mode == "sync" and self.profiler.uses_cuda:
            torch.cuda.synchronize(self.profiler.device)
        if self.use_cuda and self.profiler.cuda_mode in ("event", "sync") and self.profiler.uses_cuda:
            self.cuda_start = torch.cuda.Event(enable_timing=True)
            self.cuda_end = torch.cuda.Event(enable_timing=True)
            self.cuda_start.record()
        self.cpu_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cuda_end is not None:
            self.cuda_end.record()
        if self.use_cuda and self.profiler.cuda_mode == "sync" and self.profiler.uses_cuda:
            torch.cuda.synchronize(self.profiler.device)
        elapsed_ms = (time.perf_counter() - self.cpu_start) * 1000.0
        self.profiler.record(self.name, elapsed_ms, self.cuda_start, self.cuda_end)
        return False


class StageProfiler:
    def __init__(self, enabled=True, cuda_mode="event", device=None):
        self.enabled = bool(enabled)
        self.cuda_mode = cuda_mode
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.uses_cuda = (
            self.enabled
            and self.cuda_mode in ("event", "sync")
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self._lock = threading.Lock()
        self._cpu_stats = {}
        self._pending_cuda_events = []
        self._window_start_time = time.perf_counter()
        self._window_start_step = 0

    def stage(self, name, use_cuda=None):
        if not self.enabled:
            return nullcontext()
        if use_cuda is None:
            use_cuda = name in DEFAULT_CUDA_STAGES
        return _StageTimer(self, name, use_cuda=bool(use_cuda))

    def record(self, name, elapsed_ms, cuda_start=None, cuda_end=None):
        if not self.enabled:
            return
        with self._lock:
            self._add_stat(self._cpu_stats, name, elapsed_ms)
            if cuda_start is not None and cuda_end is not None:
                self._pending_cuda_events.append((name, cuda_start, cuda_end))

    def has_data(self):
        with self._lock:
            return bool(self._cpu_stats or self._pending_cuda_events)

    def snapshot(self, global_step, metadata=None):
        if not self.enabled:
            return None

        now = time.perf_counter()
        with self._lock:
            cpu_stats = self._cpu_stats
            pending_cuda_events = self._pending_cuda_events
            window_start_time = self._window_start_time
            window_start_step = self._window_start_step
            self._cpu_stats = {}
            self._pending_cuda_events = []
            self._window_start_time = now
            self._window_start_step = int(global_step)

        cuda_stats = {}
        for name, cuda_start, cuda_end in pending_cuda_events:
            try:
                cuda_end.synchronize()
                elapsed_ms = cuda_start.elapsed_time(cuda_end)
            except RuntimeError:
                continue
            self._add_stat(cuda_stats, name, elapsed_ms)

        stages = {}
        for name in sorted(set(cpu_stats) | set(cuda_stats)):
            stage_stats = self._format_stats(cpu_stats.get(name)) or self._zero_stats()
            cuda_stage_stats = self._format_stats(cuda_stats.get(name))
            if cuda_stage_stats is not None:
                stage_stats.update(
                    {
                        "cuda_total_ms": cuda_stage_stats["total_ms"],
                        "cuda_mean_ms": cuda_stage_stats["mean_ms"],
                        "cuda_max_ms": cuda_stage_stats["max_ms"],
                    }
                )
            stages[name] = stage_stats

        row = {
            "global_step": int(global_step),
            "window_steps": int(global_step) - int(window_start_step),
            "wall_time_s": now - window_start_time,
            "cuda_mode": self.cuda_mode,
            "stages": stages,
        }
        if metadata:
            row.update(metadata)
        return row

    @staticmethod
    def _add_stat(stats, name, elapsed_ms):
        entry = stats.setdefault(name, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
        entry["count"] += 1
        entry["total_ms"] += float(elapsed_ms)
        entry["max_ms"] = max(entry["max_ms"], float(elapsed_ms))

    @staticmethod
    def _format_stats(stats):
        if stats is None:
            return None
        count = int(stats["count"])
        total_ms = float(stats["total_ms"])
        return {
            "count": count,
            "total_ms": total_ms,
            "mean_ms": total_ms / count if count else 0.0,
            "max_ms": float(stats["max_ms"]),
        }

    @staticmethod
    def _zero_stats():
        return {"count": 0, "total_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}


class ProfileJSONLWriter:
    def __init__(self, path, profiler):
        self.path = path
        self.profiler = profiler
        self._lock = threading.Lock()
        self._file = open(path, "w", encoding="utf-8") if profiler.enabled else None

    def write_snapshot(self, global_step, metadata=None, force=False):
        if self._file is None:
            return None
        if not force and not self.profiler.has_data():
            return None

        row = self.profiler.snapshot(global_step, metadata=metadata)
        if row is None:
            return None
        with self._lock:
            self._file.write(json.dumps(row) + "\n")
            self._file.flush()
        return row

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
