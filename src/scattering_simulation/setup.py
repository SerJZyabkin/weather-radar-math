#!/usr/bin/env python3
from distutils.core import setup
import os


package_names = list()
for directory_info in os.walk('scattering_simulation'):
    if directory_info[0][-len('__package__'):] != '__pycache__':
        package_names.append(directory_info[0])

setup(
    name='scattering_simulation',
    version='0.1dev',
    packages=package_names,
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description="Набор процедур моделирования радиолокационного сигнала, отраженного различных типов гидрометеоров",
    long_description=open('README.rst').read(),
)