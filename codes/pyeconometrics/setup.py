from setuptools import setup

setup(
    name='pyeconometrics',
    version='1.0.2',
    description='Econometrics Models for Python',
    long_description=open('README.md').read(),
    author='Nicolas HENNETIER',
    author_email='nicolashennetier2@gmail.com',
    packages=['pyeconometrics'],
    requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'sklearn']
)