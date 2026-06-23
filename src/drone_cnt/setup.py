"""Package definition for drone_cnt."""

from glob import glob

from setuptools import find_packages, setup

package_name = "drone_cnt"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="shaw",
    maintainer_email="shawwong@yeah.net",
    description="OSQP MPC and MPCC controllers producing PX4 CTBR commands.",
    license="Apache-2.0",
    extras_require={"test": ["pytest", "PyYAML"]},
    entry_points={
        "console_scripts": [
            "drone_cnt_mpc = drone_cnt.base:main_mpc",
            "drone_cnt_mpcc = drone_cnt.base:main_mpcc",
        ]
    },
)
