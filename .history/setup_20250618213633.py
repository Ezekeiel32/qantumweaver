from setuptools import setup, find_packages

setup(
    name="tetrazpe",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
    ],
    python_requires=">=3.8",
) 