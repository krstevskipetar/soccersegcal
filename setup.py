from pathlib import Path
from setuptools import setup, find_packages


with open("requirements.txt") as req_file:
    requirements = [line.strip() for line in req_file if line.strip() and not line.startswith("#")]


setup(
    name="soccersegcal",
    version="1.0.0",
    description="Soccer Field Segmentation and Camera Calibration",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    url="https://github.com/Spiideo/soccersegcal/",
    author="Hakan Ardo",
    author_email="hakan.ardo@spiideo.com",
    entry_points={
        "console_scripts": ["soccersegcal-infer=soccersegcal.inference:main"],
    },
    zip_safe=False,
)

