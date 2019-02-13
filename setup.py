#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="symbolic-pymc3",
    version="0.0.1",
    install_requires=[
        'theano',
        'pymc3',
        'pymc4',
        'scipy',
        'tf-nightly',
        'tfp-nightly',
        'kanren',
        'multipledispatch',
        'unification',
        'sympy',
        'toolz',
    ],
    packages=find_packages(exclude=['tests']),
    tests_require=[
        'pytest'
    ],
    author="Brandon T. Willard",
    author_email="brandonwillard+spymc3@gmail.com",
    long_description="""Symbolic mathematics extensions for Theano and PyMC3.""",
    license="LGPL-3",
    url="https://github.com/brandonwillard/symbolic-pymc3",
    platforms=['any'],
    python_requires='>=3.6',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: DFSG approved",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
    ]
)
