from glob import glob

from setuptools import find_packages, setup

package_name = 'pet_drive'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Fan Zhu',
    maintainer_email='fzhu.admin@icloud.com',
    description='Driving the TurtleBot 4: the safety layer every drive command goes through',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'safety = pet_drive.safety_node:main',
        ],
    },
)
