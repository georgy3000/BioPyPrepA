# Общие настройки
BLOCK_SIZE = (256, 1, 1)


def compute_grid(n, block=BLOCK_SIZE):
    return ((n + block[0] - 1) // block[0], 1)
