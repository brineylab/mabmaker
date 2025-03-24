# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import os

from setuptools import find_packages, setup

# the actual __version__ is read from version.py, not assigned directly
# this causes the linter to freak out when we mention __version__ in setup()
# to fix that, we fake assign it here
__version__ = None

# read version
version_file = os.path.join(os.path.dirname(__file__), "mabmaker", "version.py")
with open(version_file) as version:
    exec(version.read())

# read requirements
with open("requirements.txt") as reqs:
    requirements = reqs.read().splitlines()

# read long description
with open("README.md", encoding="utf-8") as readme:
    long_description = readme.read()

setup(
    name="mabmaker",
    version=__version__,
    author="Bryan Briney",
    author_email="briney@scripps.edu",
    description="mAb maker: in silico antibody design.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brineylab/mabmaker",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "mabmaker=mabmaker.scripts.mabmaker:cli",
        ]
    },
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.10",
)
