from setuptools import setup, find_packages

setup(
    name='DL-reservoir-modeling',
    version='0.1.0',
    # find source code as package
    packages=find_packages(include=['src', 'src.*']), 
    # install dependencies
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'jupyter',
        'torch',
        'tqdm'
    ] 

)