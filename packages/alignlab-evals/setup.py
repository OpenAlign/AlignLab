from setuptools import setup, find_packages

setup(
    name="alignlab-evals",
    version="0.1.0",
    author="AlignLab Team",
    author_email="team@alignlab.org",
    description="Adapters for external evaluation frameworks",
    long_description="Integration adapters for lm-eval-harness, OpenAI Evals, JailbreakBench, and HarmBench",
    long_description_content_type="text/markdown",
    url="https://github.com/alignlab/alignlab",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "alignlab-core>=0.1.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "lm-eval": [
            "lm-eval>=0.4.0",
        ],
        "openai-evals": [
            "openai-evals>=0.2.0",
        ],
        "jailbreakbench": [
            # JailbreakBench dependencies would go here
        ],
        "harmbench": [
            # HarmBench dependencies would go here
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "isort>=5.0",
            "mypy>=0.900",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

