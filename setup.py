from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="mlgame3d",
    version="0.8.0",
    author="PAIA",
    author_email="service@paia-tech.com",
    description="A framework for playing Unity games with Python agents using ML-Agents Environment.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PAIA-Playful-AI-Arena/MLGame3D",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
)
