#!/usr/bin/env python
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='EIReg',
    version='0.2',
    author='Lucas Oyarzun',
    description='Tabular interpretable regression with embedded interpreter',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    packages=['EIReg'],
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
        'dill'
    ]
)
