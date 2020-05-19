from setuptools import setup, find_packages
import imp

version = imp.load_source('yata.version', 'waveglow_vocoder/version.py')
description='A vocoder that can convert audio to Mel-Spectrogram and reverse with WaveGlow, all on GPU(if avaliable).'

with open('README.md') as file:
    long_description = file.read()

setup(
    name='waveglow_vocoder',
    version=version.version,
    description=description,
    author='HudsonHuang',
    author_email='h.zy@mail.scut.edu.cn',
    url='http://github.com/HudsonHuang/waveglow_vocoder',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords='deep learning, audio processing, machine learning',
    license='BSD-3',
    install_requires=[
    ]
)