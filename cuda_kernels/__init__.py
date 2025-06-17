import os
from pycuda.compiler import SourceModule


def load_kernel(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "r") as f:
        source = f.read()
    return SourceModule(source)


# Компиляция всех ядер
mod_preprocess = load_kernel("preprocess.cu")

# Доступ к функциям
filter_low_quality_cuda = mod_preprocess.get_function("filter_quality_cuda")
count_lines_cuda = mod_preprocess.get_function("count_lines_cuda")
normalize_cuda = mod_preprocess.get_function("normalize_cuda")
trim_primers_cuda = mod_preprocess.get_function("trim_primers_cuda")
trim_adapters_cuda = mod_preprocess.get_function("trim_adapters_cuda")
filter_by_length_cuda = mod_preprocess.get_function("filter_by_length_cuda")
# align_sequences = mod_preprocess.get_function("align_sequences")

