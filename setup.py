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
        #        'tensorflow-gpu',
        #        'click',
        #        'cachey',
        #        'napari',
        #        'dask',
        #        'zarr',
        #        'keras',
        #        'dtcwt',
        #        'cupy-cuda100',
        #        'gputools',
        #        'numexpr'
    ],

    entry_points='''
        [console_scripts]
        dexp=dexp.cli.main:cli
    ''',
)
