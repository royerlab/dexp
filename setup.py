#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import atexit
import io
import os
import sys
from os.path import exists, join
from shutil import rmtree

from setuptools import find_packages, setup, Command
from setuptools.command.install import install


# Package meta-data.
NAME = 'dexp'
DESCRIPTION = 'Light-sheet Dataset EXploration and Processing'
URL = 'https://github.com/royerlab/dexp'
EMAIL = 'jordao.bragantini@czbiohub.org, ahmetcan.solak@czbiohub.org, bin.yang@czbiohub.org, loic.royer@czbiohub.org'
AUTHOR = 'Jordao Bragantini, Ahmet Can Solak, Bin Yang, Loic A Royer'
REQUIRES_PYTHON = '>=3.7.0'

from datetime import datetime
now = datetime.now()
seconds_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
minutes_since_midnight = int(seconds_since_midnight//60)
ten_minutes_since_midnight = int(seconds_since_midnight//600)
VERSION = datetime.today().strftime('%Y.%m.%d')+f'.{minutes_since_midnight}'
print(f"Version: {VERSION}")

CUPY_VERSION = '9.1.0'

# What packages are required for this module to be executed?
REQUIRED = [
    'numpy>=1.20',
    'scipy',
    'numba',
    'numcodecs',
#    'cucim',  # optional
    'scikit-image',
    'tifffile',
    'numexpr',
    'joblib',
    'pytest',
    'click==7.1.2',
    'dask',
    'dask-image',
    'dask-cuda',
    'distributed',
    'zarr',
    'cachey',
    'gputil',
    'gpustat',
    'gputools',
    'arbol',
    'colorcet',
    'PyYAML',
    'cairocffi',
    'blosc',
    'seaborn',
    'ome-zarr',
    'pyotf',
]

# What packages are optional?
EXTRAS = {
    'source': [f'cupy=={CUPY_VERSION}', ],
    'cuda112': [f'cupy-cuda112=={CUPY_VERSION}', ],
    'cuda111': [f'cupy-cuda111=={CUPY_VERSION}', ],
    'cuda110': [f'cupy-cuda110=={CUPY_VERSION}', ],
    'cuda102': [f'cupy-cuda102=={CUPY_VERSION}', ],
    'cuda101': [f'cupy-cuda101=={CUPY_VERSION}', ],
    'cuda100': [f'cupy-cuda100=={CUPY_VERSION}', ],
    'color': ['colorama', 'ansicolors',],
    'napari': ['napari[all]',]
}





# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        # install twine if not
        os.system('pip install -y twine --dev')

        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status(f'Uploading version `{about["__version__"]}` of the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()



# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    entry_points={
    'console_scripts': ['dexp=dexp.cli.dexp_main:cli',
                        'video=dexp.cli.video_main:cli',
                        'install=dexp.cli.install_main:cli']
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='BSD 3-Clause',
    license_file='LICENSE.txt',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Environment :: GPU :: NVIDIA CUDA :: 11.1',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand
    },
)



