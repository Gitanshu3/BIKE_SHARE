from setuptools import setup, find_packages

setup(
    name="bike_sharing_model",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "strictyaml",
        "pytest",
    ],
    entry_points={
        "console_scripts": [
        ],
    },
)