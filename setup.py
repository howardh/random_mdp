from setuptools import setup, find_packages

setup(name='random_mdp',
    version='0.0.1',
    install_requires=[
        'numpy',
    ],
    extras_require={
        'dev': ['pytest','pytest-cov','pdoc','flake8','autopep8'],
    },
    packages=find_packages()
)
