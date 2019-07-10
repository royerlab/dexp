from setuptools import setup

setup(
    name='dexp',
    version='0.0.1',
    packages=['dexp', 'dexp.cli', 'dexp.demo', 'dexp.datasets', 'dexp.datasets.demo'],
    url='',
    license='',
    author='royer',
    author_email='',
    description='',

    install_requires=[
        'click',
        'napari',
        'dask',
        'zarr',
        'numcodecs',
        'numpy',
        'tifffile',
        'cachey'
    ],

    entry_points='''
        [console_scripts]
        dexp=dexp.cli.cli:cli
    ''',
)
