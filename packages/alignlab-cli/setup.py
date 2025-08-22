"""
Setup script for alignlab-cli package.
"""

from setuptools import setup, find_packages

setup(
    name="alignlab-cli",
    version="0.1.0",
    author="AlignLab Team",
    author_email="team@alignlab.org",
    description="Command-line interface for AlignLab",
    long_description="CLI tools for the AlignLab alignment evaluation framework",
    long_description_content_type="text/markdown",
    url="https://github.com/alignlab/alignlab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "alignlab-core>=0.1.0",
    ],
    entry_points={
        "console_scripts": [
            "alignlab=alignlab_cli.main:main",
        ],
    },
)

