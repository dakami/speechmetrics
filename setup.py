# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

setup(
    name="speechmetrics",
    version="1.0.2dmk",
    packages=find_packages(),

    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'resampy',
        'pystoi',
        'museval',
        'librosa',
        'lazy_import',
        'tf-nightly@https://storage.googleapis.com/dakami-canyouhearme-no/tf_nightly-2.1.0-cp37-cp37m-linux_x86_64.whl',
        'tflite_runtime@https://storage.googleapis.com/dakami-canyouhearme-no/tflite_runtime-2.1.0-cp37-cp37m-linux_x86_64.whl',
        'snoop'
    ],
    extras_require={
        'gpu': ['tensorflow-gpu==2.0.0', 'librosa'],
    },
    include_package_data=True
)

#https://github.com/inoryy/tensorflow-optimized-wheels/releases/download/v2.1.0/tensorflow-2.1.0-cp37-cp37m-linux_x86_64.whl
