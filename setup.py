"""Python setup.py for xLSTM-Mixer package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("xlstm_mixer", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="xlstm_mixer",
    version=read("xlstm_mixer", "VERSION"),
    description="xLSTM-MIXER for forecasting", 
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=[".docker", ".devcontainer", ".github"]),
    install_requires=read_requirements(
        ["requirements.txt", "lightning-requirements.txt"]
    ),
    entry_points={"console_scripts": ["xlstm_mixer = xlstm_mixer.__main__:main"]},
)
