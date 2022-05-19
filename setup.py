import setuptools

# description
with open('README.md') as f:
    long_description = f.read()
    
# requirements
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()
    
# version
import afc.__init__ as base
version = base.__version__

setuptools.setup(
    name="AFC",
    version=version,
    author="Gehbauer, Christoph",
    description="Advanced Facade Control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LBNL-ETA/AFC",
    project_urls={
        "Bug Tracker": "hhttps://github.com/LBNL-ETA/AFC/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=install_requires
)