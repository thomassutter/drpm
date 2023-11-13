from setuptools import setup
from setuptools import find_packages

DIR_MVHG = "/path/to/mvhg/git/repo"

setup(
    name="drpm",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        f"mvhg @ file://localhost/{DIR_MVHG}#egg=mvhg",
    ],
    extras_require={
        "pt": ["torch"],
        "tf": ["tensorflow"],
        "tf_gpu": ["tensorflow-gpu"],
    },
)
