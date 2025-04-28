"""
Setup file for the Advanced Fenestration Controller.
"""

import os
import sys
import json
import setuptools

# description
with open('README.md', 'r', encoding='utf8') as f:
    long_description = f.read()

# requirements
with open('requirements.txt', 'r', encoding='utf8') as f:
    install_requires = f.read().splitlines()

# version
with open('afc/__init__.py', 'r', encoding='utf8') as f:
    version = json.loads(f.read().split('__version__ = ')[1])

setuptools.setup(
    name="afc",
    version=version,
    author="Gehbauer, Christoph",
    description="Advanced Facade Control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license_files = ['license.txt'],
    url="https://github.com/LBNL-ETA/AFC",
    project_urls={
        "Bug Tracker": "https://github.com/LBNL-ETA/AFC/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=['afc'],
    package_data={'': ['*.txt', '*.md'], 
                  'afc': ['glare/*',
                          'radiance/*',
                          'utility/*',
                          'resources/*',
                          'resources/*/*',
                          'resources/*/*/*',
                          'resources/*/*/*/*',
                          'resources/*/*/*/*/.*',
                          'resources/*/*/*/*/*']},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=install_requires
)
