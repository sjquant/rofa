import io

from setuptools import setup

with io.open("README.rst", mode='r', encoding='utf-8') as rm:
    long_description = rm.read()

setup(
    name='rofa',
    version='0.1.3',
    author='SJQuant',
    license="MIT",
    author_email='seonujang92@gmail.com',
    description=(
        '**Rofa ** is abbreviation for Robust Factor'
    ),
    url='https://github.com/sjquant/rofa',
    long_description=long_description,
    packages=['rofa'],
    keywords=['quant', 'investment', 'factor investing', 'backtest'],
    install_requires=['pandas>=0.24.0', 'seaborn>=0.9.0'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
