mport numpy as np
import pycuda.driver as cuda
from .cuda_kernels import normalize_kernel, filter_low_quality, align_sequences
from . import utils

def filter_quality(seqs: np.ndarray, qual: np.ndarray, threshold: int = 20) -> np.ndarray:
    seqs = seqs.astype(np.byte)
    qual = qual.astype(np.int32)
    n = seqs.size

    seqs_gpu = cuda.mem_alloc(seqs.nbytes)
    qual_gpu = cuda.mem_alloc(qual.nbytes)

    cuda.memcpy_htod(seqs_gpu, seqs)
    cuda.memcpy_htod(qual_gpu, qual)

    block = (256, 1, 1)
    grid = ((n + block[0] - 1) // block[0], 1)

    filter_low_quality(seqs_gpu, qual_gpu, np.int32(n), np.int32(threshold), block=block, grid=grid)

    result = np.empty_like(seqs)
    cuda.memcpy_dtoh(result, seqs_gpu)
    return result.tobytes().decode("ascii")
