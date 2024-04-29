from setuptools import find_packages, setup

install_requires = ["onnxruntime", "torch>=1.9", "sentencepiece", "huggingface-hub", "omegaconf", "numpy"]

setup(
    name="punctuators",
    version="0.0.6",
    description="Punctuators and such",
    author="Shane",
    author_email="shane.carroll@utsa.edu",
    python_requires=">=3.6",
    packages=find_packages(),
    install_requires=install_requires,
)
