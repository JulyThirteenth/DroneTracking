"""Setuptools definition for the drone_ccm ROS 2 package."""

from glob import glob
from pathlib import Path

from setuptools import find_packages, setup


PACKAGE_NAME = "drone_ccm"

DATA_FILES = [
    (
        "share/ament_index/resource_index/packages",
        ["resource/" + PACKAGE_NAME],
    ),
    ("share/" + PACKAGE_NAME, ["package.xml", "README.md"]),
    ("share/" + PACKAGE_NAME + "/config", glob("config/*.yaml")),
    ("share/" + PACKAGE_NAME + "/launch", glob("launch/*.launch.py")),
]
MODEL_NAMES = ("neu_ccm_practical.pt", "neu_ego_ccm_active.pt")
MODEL_FILES = [str(Path("ctbr_cnt") / name) for name in MODEL_NAMES]
DATA_FILES.append(("share/" + PACKAGE_NAME + "/models", MODEL_FILES))


setup(
    name=PACKAGE_NAME,
    version="0.9.0",
    packages=find_packages(exclude=("test",)),
    data_files=DATA_FILES,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="shaw",
    maintainer_email="shawwong@yeah.net",
    description="Standard and ego-centric CCM velocity-attitude tracking.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "drone_ccm_controller = drone_ccm.controller_node:main",
            "drone_ccm_reference = drone_ccm.reference_node:main",
        ],
    },
)
