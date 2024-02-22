"""A profiling runner for JAX workers"""

import jax
from acme import core
from acme.utils import signals


class ProfilingServerRunner(core.Worker):
  """A runner instance which wraps the run() with a JAX profiling server."""

  def __init__(self, wrapped: core.Worker, port: int = 9999):
    self._wrapped = wrapped
    self._port = port

  def run(self):
    jax.profiler.start_server(self._port)
    with signals.runtime_terminator(jax.profiler.stop_server):
      self._wrapped.run()

  def __getattr__(self, name):
    if name=="run":
      return self.run 
    return getattr(self._wrapped, name)
  

class ProfilingTraceRunner(core.Worker):
  """A runner instance which collects JAX profiler trace from run()."""

  def __init__(self, wrapped: core.Worker, log_dir: str = "/tmp/tensorboard"):
    self._wrapped = wrapped
    self._log_dir = log_dir

  def run(self):
    jax.profiler.start_trace(self._log_dir)
    with signals.runtime_terminator(jax.profiler.stop_trace):
      self._wrapped.run()

  def __getattr__(self, name):
    if name=="run":
      return self.run 
    return getattr(self._wrapped, name)