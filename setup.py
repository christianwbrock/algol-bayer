"""A setuptools based setup module.
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    required = f.read().splitlines()


setup(
    name='bayer',
    version='1.0.0a1',
    description='bayer-masked image reduction package',
    long_description=long_description,
    author='Christian W. Brock',
    license='BSD',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License'
    ],
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': [
            'bayer_histogram=bayer.display_histogram:main',
            'bayerfast=bayer.bayerfast:main'
        ]
    }
)
