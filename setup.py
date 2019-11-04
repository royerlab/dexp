from setuptools import setup

setup(
    name='dexp',
    version='0.0.1',
    packages=['dexp', 'dexp.cli', 'dexp.datasets', 'dexp.datasets.demo'],
    url='',
    license='',
    author='royer',
    author_email='',
    description='',

    install_requires=[
        'click',
        'cachey',
        'napari',
        'numpy',
        'dask',
        'zarr',
        'tensorflow-gpu',
        'keras',
        'dtcwt',
        'cupy-cuda100',
        'gputools',
        'numexpr'

    ],

    entry_points='''
        [console_scripts]
        dexp=dexp.cli.main:cli
    ''',
)
