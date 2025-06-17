from setuptools import setup, find_packages

setup(
    name="BioPyPrepA",  # Имя библиотеки
    version="0.1.1",     # Первая версия
    description="GPU-ускоренная библиотека для препроцессинга биологических последовательностей",
    author="Georgy3000",  # Твоё имя или ник
    url="https://github.com/georgy3000/BioPyPrepA",
    packages=find_packages(),  # Автоматически находит все пакеты
    include_package_data=True,  # включает не-Python файлы, если указан MANIFEST.in
    install_requires=[
        "numpy",
        "pycuda",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
