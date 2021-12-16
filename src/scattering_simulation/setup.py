#!/usr/bin/env python
from distutils.core import setup

setup(
    name='scattering_simulation',
    version='0.1dev',
    packages=['scattering_simulation', 'scattering_simulation.visualization'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description="Набор процедур моделирования радиолокационного сигнала, отраженного различных типов гидрометеоров",
    long_description=open('README.rst').read(),
)