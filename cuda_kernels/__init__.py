import os
from pycuda.compiler import SourceModule

def load_kernel(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, 'r') as f:
        source = f.read()
    return SourceModule(source)

# Компиляция всех ядер
mod_preprocess = load_kernel("preprocess.cu")

# Доступ к функциям
filter_low_quality = mod_preprocess.get_function("filter_quality")

