from setuptools import setup, find_packages

setup(
    name="posformer",
    version="0.1.0",
    description="PosFormer: A deep learning model for ...",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "pytorch-lightning",
        "numpy",
        "typer",
        "PyYAML",
        # Add other dependencies from environment.yml as needed
    ],
)
