"""
Setup script for alignlab-guards package.
"""

from setuptools import setup, find_packages

setup(
    name="alignlab-guards",
    version="0.1.0",
    author="AlignLab Team",
    author_email="team@alignlab.org",
    description="Guard models and rule engines for AlignLab",
    long_description="Safety filtering and guard models for the AlignLab alignment evaluation framework",
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
        "alignlab-core>=0.1.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)

