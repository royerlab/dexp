from setuptools import setup

setup(
    name='dexp',
    version='0.0.1',
    packages=['dexp', 'dexp.cli'],
    url='',
    license='',
    author='royer',
    author_email='',
    description='',

    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html"
    ],

    install_requires=[
               'numpy',
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
               'cupy-cuda111==9.0.0b3',

    ],

    #
    # pip install numpy --upgrade
    # pip install scipy --upgrade
    # pip install numba --upgrade
    # pip install numcodecs --upgrade
    # pip install scikit-image --upgrade
    # pip install tifffile --upgrade
    # pip install numexpr --upgrade
    # pip install joblib --upgrade
    # pip install pytest --upgrade
    # pip install graphviz --upgrade
    # pip install click --upgrade
    # pip install dask --upgrade
    # pip install dask-image --upgrade
    # pip install distributed --upgrade
    # pip install zarr --upgrade
    #
    # pip install napari[all] --upgrade
    # pip install cachey --upgrade
    # pip install spimagine --upgrade
    # pip install gputil gpustat --upgrade
    # pip install arbol colorama ansicolors --upgrade
    # pip install simple-zarr-server requests aiohttp --upgrade
    # pip install colorcet --upgrade
    # pip install python-telegram-bot --upgrade
    # pip install PyYAML --upgrade
    # pip install pycairo==1.11.1
    #
    # pip install cupy-cuda111==9.0.0b3 --no-cache-dir
    #



    entry_points='''
        [console_scripts]
        dexp=dexp.cli.dexp_main:cli
        video=dexp.cli.video_main:cli
    ''',
)
