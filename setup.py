# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

setup(
    name="speechmetrics",
    version="1.0",
    packages=find_packages(),

    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'resampy',
        'pystoi',
        'museval',
        'tf-nightly',
        'librosa'
    ],
    include_package_data=True
)
