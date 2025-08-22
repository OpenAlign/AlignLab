"""
Setup script for alignlab-core package.
"""

from setuptools import setup, find_packages

# Read README if it exists, otherwise use a default description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Core library for alignment evaluation framework"

setup(
    name="alignlab-core",
    version="0.1.0",
    author="AlignLab Team",
    author_email="team@alignlab.org",
    description="Core library for alignment evaluation framework",
    long_description=long_description,
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
        "datasets>=2.14.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
        "openai>=1.0.0",
        "requests>=2.28.0",
        "weasyprint>=60.0",  # For PDF generation
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
        "vertex": [
            "google-cloud-aiplatform>=1.35.0",
        ],
        "vllm": [
            "vllm>=0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "alignlab=alignlab_core.cli:main",
        ],
    },
)
