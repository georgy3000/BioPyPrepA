// Удаление низкокачественных оснований
__global__ void filter_quality_cuda(char *seqs, int *qual, int len, int threshold) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len && qual[idx] < threshold) {
        seqs[idx] = 'N';  // заменяем на неизвестное основание
    }
}

// Подсчёт количества ридов
__global__ void  count_lines_cuda(const char* data, int size, int* line_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int local_count = 0;
    for (int i = idx; i < size; i += stride) {
        if (data[i] == '\n') {
            local_count++;
        }
    }

    atomicAdd(line_count, local_count);
}

//обрезка 
__global__ void trim_primers_cuda(
    char* seqs,         // все последовательности в один массив
    int* seq_offsets,   // смещения начала каждой последовательности
    int* seq_lengths,   // длины каждой последовательности
    const char* primer, // строка праймера
    int primer_len
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= gridDim.x * blockDim.x) return;

    int start = seq_offsets[idx];
    int len = seq_lengths[idx];

    if (len < primer_len) return;

    bool match = true;
    for (int i = 0; i < primer_len; i++) {
        if (seqs[start + i] != primer[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        // сдвиг влево
        for (int i = 0; i < len - primer_len; i++) {
            seqs[start + i] = seqs[start + primer_len + i];
        }
        seq_lengths[idx] -= primer_len;
    }
}

__global__ void trim_adapters_cuda(
    char* seqs,        // all sequences in one array
    int* seq_offsets,  // offsets of the start of each sequence
    int* seq_lengths,  // lengths of each sequence
    const char* adapter, // adapter string
    int adapter_len
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= gridDim.x * blockDim.x) return;

    int start = seq_offsets[idx];
    int len = seq_lengths[idx];

    if (len < adapter_len) return;

    // Check for adapter at the end
    bool match = true;
    for (int i = 0; i < adapter_len; i++) {
        if (seqs[start + len - adapter_len + i] != adapter[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        // Trim adapter from the end
        seq_lengths[idx] -= adapter_len;
    }
}

// Нормализация значений флуоресценции или частот
__global__ void normalize_cuda(float *data, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        data[idx] = data[idx] / 100.0f;  // условная нормализация
    }
}

//обрезка по длине
__global__ void filter_by_length_cuda(
    int* lengths,      // длины ридов
    int* keep_flags,   // флаги: 1 — оставить, 0 — отфильтровать
    int min_length,
    int max_length,
    int n
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    keep_flags[idx] = ((lengths[idx] >= min_length)&&(lengths[idx] <= max_length)) ? 1 : 0;
}

