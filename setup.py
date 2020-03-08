from setuptools import setup, find_packages


setup(
    name = 'nbodies',
    version = '0.1.0',
    install_requires = ['numpy', 'scipy', 'matplotlib', 'progress'],
    packages = find_packages()
)
