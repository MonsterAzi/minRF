import torch
import time
import torch.nn.functional as F

@torch.jit.script
def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """
  Computes the softmax of a tensor along a specified dimension.

  Args:
    x: Input tensor.
    dim: The dimension along which to compute the softmax.

  Returns:
    A tensor with the same shape as x, containing the softmax values.
  """
  e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])  # Subtract max for numerical stability
  return e_x / e_x.sum(dim=dim, keepdim=True)

@torch.jit.script
def stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """
  Computes the StableMax of a tensor along a specified dimension.

  Args:
    x: Input tensor.
    dim: The dimension along which to compute the StableMax.

  Returns:
    A tensor with the same shape as x, containing the StableMax values.
  """
  s_x = torch.where(x >= 0, x + 1, 1 / (1 - x))
  return s_x / s_x.sum(dim=dim, keepdim=True)

# Example usage and simple benchmarking
x = torch.randn(1024, 2048, device="cuda")  # Example large tensor on GPU

# Time and memory for F.softmax
start_time = time.time()
y_f = F.softmax(x, dim=-1)
end_time = time.time()
print(f"F.softmax: Time = {end_time - start_time:.4f} s, Memory allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
del y_f
torch.cuda.empty_cache()

# Time and memory for memory-efficient my_softmax
torch.cuda.reset_peak_memory_stats()
start_time = time.time()
y_my = my_softmax(x, dim=-1)
end_time = time.time()
print(f"my_softmax: Time = {end_time - start_time:.4f} s, Memory allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
del y_my
torch.cuda.empty_cache()

# Time and memory for memory-efficient stablemax
torch.cuda.reset_peak_memory_stats()
start_time = time.time()
y_stable = stablemax(x, dim=-1)
end_time = time.time()
print(f"stablemax: Time = {end_time - start_time:.4f} s, Memory allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
del y_stable
torch.cuda.empty_cache()