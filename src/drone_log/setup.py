"""Package definition for drone_log."""

from glob import glob

from setuptools import find_packages, setup


setup(
    name="drone_log",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/drone_log"]),
        ("share/drone_log", ["package.xml", "README.md"]),
        ("share/drone_log/config", glob("config/*.yaml")),
        ("share/drone_log/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="shaw",
    maintainer_email="shawwong@yeah.net",
    description="Independent CSV flight logger for MPC and MPCC tracking.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={"console_scripts": ["drone_log = drone_log.node:main"]},
)
