from glob import glob

from setuptools import find_packages, setup

package_name = 'drone_ref'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/waypoints', glob('waypoints/*.txt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shaw',
    maintainer_email='shawwong@yeah.net',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "drone_ref_mpc = drone_ref.mpc:main",
            "drone_ref_mpcc = drone_ref.mpcc:main",
        ],
    },
)
