import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
import mlgame3d

VERSION = mlgame3d.__version__

class VerifyVersionCommand(install):
    """
    Custom command to verify that the git tag is the expected one for the release.
    Originally based on https://circleci.com/blog/continuously-deploying-python-packages-to-pypi-with-circleci/
    This differs slightly because our tags and versions are different.
    """

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("GITHUB_REF", "NO GITHUB TAG!").replace("refs/tags/", "")

        if tag != VERSION:
            info = "Git tag: {} does not match the version of this app: {}".format(
                tag, VERSION
            )
            sys.exit(info)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="mlgame3d",
    version="0.1.0",
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
