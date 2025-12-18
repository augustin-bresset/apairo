from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# read the contents of the README file
from os import path
this_directory = path.abspath(path.dirname(__file__))

setup(
    name='Data Apairo',
    version='0.1dev',
    description='Extemsible Framework to manage data',
    author='Augustin Bresset',
    author_email='augustin.bresset@gmail.com',
    # packages=['src',],
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
