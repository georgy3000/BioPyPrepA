import numpy as np
import pycuda.driver as cuda
from .cuda_kernels import (
    filter_low_quality_cuda,
    count_lines_cuda,
    trim_primers_cuda,
    trim_adapters_cuda,
    filter_by_length_cuda,
)
from utils import BLOCK_SIZE, compute_grid


# Подсчёт количества ридов
def count_reads(fastq_path: str) -> int:
    with open(fastq_path, "rb") as f:
        data = f.read()

    byte_data = np.frombuffer(data, dtype=np.uint8)
    n = len(byte_data)
    size = np.int32(n)

    # GPU память
    data_gpu = cuda.mem_alloc(byte_data.nbytes)
    count_gpu = cuda.mem_alloc(np.int32(0).nbytes)

    cuda.memcpy_htod(data_gpu, byte_data)
    cuda.memcpy_htod(count_gpu, np.zeros(1, dtype=np.int32))

    # параметры запуска
    grid = compute_grid(n, BLOCK_SIZE)

    count_lines_cuda(data_gpu, size, count_gpu, block=BLOCK_SIZE, grid=grid)

    result = np.zeros(1, dtype=np.int32)
    cuda.memcpy_dtoh(result, count_gpu)

    # Каждые 4 строки — один рид
    return result[0] // 4


# фильтр по качеству
def filter_quality(
    seqs: np.ndarray, qual: np.ndarray, threshold: int = 20
) -> np.ndarray:
    seqs = seqs.astype(np.byte)
    qual = qual.astype(np.int32)
    n = seqs.size

    seqs_gpu = cuda.mem_alloc(seqs.nbytes)
    qual_gpu = cuda.mem_alloc(qual.nbytes)

    cuda.memcpy_htod(seqs_gpu, seqs)
    cuda.memcpy_htod(qual_gpu, qual)

    grid = compute_grid(n, BLOCK_SIZE)

    filter_low_quality_cuda(
        seqs_gpu,
        qual_gpu,
        np.int32(n),
        np.int32(threshold),
        block=BLOCK_SIZE,
        grid=grid,
    )

    result = np.empty_like(seqs)
    cuda.memcpy_dtoh(result, seqs_gpu)

    return result.tobytes().decode("ascii")


# Обрезка праймеров
def trim_primers(fastq_path: str, primer: str) -> list[str]:
    with open(fastq_path, "rb") as f:
        lines = f.readlines()

    # Выделим только последовательности (2-я строка каждого рида)
    seq_lines = lines[1::4]
    n_reads = len(seq_lines)

    encoded_seqs = [s.strip() for s in seq_lines]
    seq_lengths = np.array([len(s) for s in encoded_seqs], dtype=np.int32)
    seq_offsets = np.array([0] + list(np.cumsum(seq_lengths[:-1])), dtype=np.int32)
    flat_seqs = np.frombuffer(b"".join(encoded_seqs), dtype=np.byte)

    primer_bytes = np.frombuffer(primer.encode("ascii"), dtype=np.byte)

    # Выделение памяти
    seqs_gpu = cuda.mem_alloc(flat_seqs.nbytes)
    offsets_gpu = cuda.mem_alloc(seq_offsets.nbytes)
    lengths_gpu = cuda.mem_alloc(seq_lengths.nbytes)
    primer_gpu = cuda.mem_alloc(primer_bytes.nbytes)

    cuda.memcpy_htod(seqs_gpu, flat_seqs)
    cuda.memcpy_htod(offsets_gpu, seq_offsets)
    cuda.memcpy_htod(lengths_gpu, seq_lengths)
    cuda.memcpy_htod(primer_gpu, primer_bytes)

    grid = compute_grid(n_reads, BLOCK_SIZE)

    trim_primers_cuda(
        seqs_gpu,
        offsets_gpu,
        lengths_gpu,
        primer_gpu,
        np.int32(len(primer)),
        block=BLOCK_SIZE,
        grid=grid,
    )

    # Получаем результат
    trimmed_seqs = np.empty_like(flat_seqs)
    new_lengths = np.empty_like(seq_lengths)

    cuda.memcpy_dtoh(trimmed_seqs, seqs_gpu)
    cuda.memcpy_dtoh(new_lengths, lengths_gpu)

    # Сборка строк
    result = []
    for i in range(n_reads):
        start = seq_offsets[i]
        end = start + new_lengths[i]
        result.append(trimmed_seqs[start:end].tobytes().decode("ascii"))

    return result


def trim_adapters(fastq_path: str, adapter: str):
    with open(fastq_path, "rb") as f:
        lines = f.readlines()

    # Выделим только последовательности (2-я строка каждого рида)
    seq_lines = lines[1::4]
    n_reads = len(seq_lines)

    encoded_seqs = [s.strip() for s in seq_lines]
    seq_lengths = np.array([len(s) for s in encoded_seqs], dtype=np.int32)
    seq_offsets = np.array([0] + list(np.cumsum(seq_lengths[:-1])), dtype=np.int32)
    flat_seqs = np.frombuffer(b"".join(encoded_seqs), dtype=np.byte)

    adapter_bytes = np.frombuffer(adapter.encode("ascii"), dtype=np.byte)

    # Выделение памяти
    seqs_gpu = cuda.mem_alloc(flat_seqs.nbytes)
    offsets_gpu = cuda.mem_alloc(seq_offsets.nbytes)
    lengths_gpu = cuda.mem_alloc(seq_lengths.nbytes)
    adapter_gpu = cuda.mem_alloc(adapter_bytes.nbytes)

    cuda.memcpy_htod(seqs_gpu, flat_seqs)
    cuda.memcpy_htod(offsets_gpu, seq_offsets)
    cuda.memcpy_htod(lengths_gpu, seq_lengths)
    cuda.memcpy_htod(adapter_gpu, adapter_bytes)

    grid = compute_grid(n_reads, BLOCK_SIZE)

    trim_adapters_cuda(
        seqs_gpu,
        offsets_gpu,
        lengths_gpu,
        adapter_gpu,
        np.int32(len(adapter)),
        block=BLOCK_SIZE,
        grid=grid,
    )

    # Получаем результат
    trimmed_seqs = np.empty_like(flat_seqs)
    new_lengths = np.empty_like(seq_lengths)

    cuda.memcpy_dtoh(trimmed_seqs, seqs_gpu)
    cuda.memcpy_dtoh(new_lengths, lengths_gpu)

    # Сборка строк
    result = []
    for i in range(n_reads):
        start = seq_offsets[i]
        end = start + new_lengths[i]
        result.append(trimmed_seqs[start:end].tobytes().decode("ascii"))

    return result


def filter_reads_by_length(
    fastq_path: str, min_length: int, max_length: int
) -> list[tuple[str, str, str]]:
    with open(fastq_path, "r") as f:
        lines = f.readlines()

    # Извлекаем все блоки ридов
    headers = lines[0::4]
    seqs = [l.strip() for l in lines[1::4]]
    quals = [l.strip() for l in lines[3::4]]

    n_reads = len(seqs)
    seq_lengths = np.array([len(s) for s in seqs], dtype=np.int32)
    keep_flags = np.zeros(n_reads, dtype=np.int32)

    # GPU память
    lengths_gpu = cuda.mem_alloc(seq_lengths.nbytes)
    flags_gpu = cuda.mem_alloc(keep_flags.nbytes)

    cuda.memcpy_htod(lengths_gpu, seq_lengths)
    cuda.memcpy_htod(flags_gpu, keep_flags)

    grid = compute_grid(n_reads, BLOCK_SIZE)

    filter_by_length_cuda(
        lengths_gpu,
        flags_gpu,
        np.int32(min_length),
        np.int32(max_length),
        np.int32(n_reads),
        block=BLOCK_SIZE,
        grid=grid,
    )

    # Получаем результат
    cuda.memcpy_dtoh(keep_flags, flags_gpu)

    # Формируем отфильтрованные риды
    result = [(headers[i], seqs[i], quals[i]) for i in range(n_reads) if keep_flags[i]]

    return result
