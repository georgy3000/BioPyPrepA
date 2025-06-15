// Удаление низкокачественных оснований
__global__ void filter_quality(char *seqs, int *qual, int len, int threshold) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len && qual[idx] < threshold) {
        seqs[idx] = 'N';  // заменяем на неизвестное основание
    }
}

// Нормализация значений флуоресценции или частот
__global__ void normalize(float *data, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        data[idx] = data[idx] / 100.0f;  // условная нормализация
    }
}
