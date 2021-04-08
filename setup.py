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
EMAIL = 'jordao.bragantini@czbiohub.org, ahmetcan.solak@czbiohub.org, loic.royer@czbiohub.org'
AUTHOR = 'Jordao Bragantini, Ahmet Can Solak, Loic A Royer'
REQUIRES_PYTHON = '>=3.8.0'

from datetime import datetime
VERSION = datetime.today().strftime('%Y.%m.%d')+'b'

CUPY_VERSION = '9.0.0b3'

# What packages are required for this module to be executed?
REQUIRED = [
    'numpy>=1.20',
    'scipy',
    'numba',
    'numcodecs',
    'scikit-image',
    'tifffile',
    'numexpr',
    'joblib',
    'pytest',
    'graphviz',
    'click',
    'dask',
    'dask-image',
    'distributed',
    'zarr',
    'napari[all]',
    'cachey',
    'gputil',
    'gpustat',
    'arbol',
    'colorama',
    'ansicolors',
    'simple-zarr-server',
    'requests',
    'aiohttp',
    'colorcet',
    'python-telegram-bot',
    'PyYAML',
    'pycairo',
]

# What packages are optional?
EXTRAS = {
    'cuda112': [f'cupy-cuda112=={CUPY_VERSION}', ],
    'cuda111': [f'cupy-cuda111=={CUPY_VERSION}', ],
    'cuda110': [f'cupy-cuda110=={CUPY_VERSION}', ],
    'cuda102': [f'cupy-cuda102=={CUPY_VERSION}', ],
    'cuda101': [f'cupy-cuda101=={CUPY_VERSION}', ],
    'cuda100': [f'cupy-cuda100=={CUPY_VERSION}', ],
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

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()

# Install additiona libraries:
class InstalLibsCommand(Command):
    """Support setup.py upload."""

    description = 'Install additional native libraries'
    user_options = [('cuda=', 'c', 'CUDA version (11.2, 11.1, 11.0, ...)'),]

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        self.cuda = None

    def finalize_options(self):
        if self.cuda is None:
            raise Exception("Parameter --cuda is missing")

    def run(self):
        from os.path import expanduser
        home = expanduser("~")
        if not exists(join(home, f'.cupy/cuda_lib/{self.cuda}')):
            self.status(f'Installing CUDNN for CUDA {self.cuda}')
            os.system(f'python -m cupyx.tools.install_library --library cudnn --cuda {self.cuda}')
            self.status(f'Installing CUTENSOR for CUDA {self.cuda}')
            os.system(f'python -m cupyx.tools.install_library --library cutensor --cuda {self.cuda}')
        else:
            self.status(f'Libraries already installed')
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

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    entry_points={
    'console_scripts': ['dexp=dexp.cli.dexp_main:cli', 'video=dexp.cli.video_main:cli']
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
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
        'upload': UploadCommand,
        'cudalibs': InstalLibsCommand
    },
)



