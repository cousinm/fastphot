from setuptools import setup, find_packages

setup(
    name = 'fastphot',
    version = '1.1.0',
    install_requires = ['numpy', 'scipy', 'matplotlib', 'progress'],
    packages = find_packages()
)
