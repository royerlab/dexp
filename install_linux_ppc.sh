# Run this to create a new env:
#conda deactivate; conda env remove --name dexp; conda create -y --name dexp python=3.8; conda activate dexp; ./install_linux.sh

conda config --add channels anaconda
conda install -y -c conda-forge numpy
conda install -y -c anaconda scipy
conda install -y -c anaconda numba
conda install -y -c anaconda numcodecs
conda install -y -c anaconda scikit-image
conda install -y -c anaconda tifffile
conda install -y -c anaconda numexpr
conda install -y -c anaconda joblib
conda install -y -c anaconda pytest
conda install -y -c anaconda graphviz
conda install -y -c anaconda click
conda install -y -c anaconda dask
conda install -y -c anaconda dask-image
conda install -y -c anaconda distributed
conda install -y -c anaconda zarr
conda install -y -c anaconda cffi
conda install -y -c anaconda cairocffi
conda install -y -c anaconda PyYAML
conda install -y -c anaconda colorcet
conda install -y -c conda-forge python-blosc
conda install -y -c conda-forge qtpy

source /apps/GCC-6.5.0/setup.GCC-6.5.0
pip install cupy==9.0.0b3 --no-cache-dir

pip install napari[all] --upgrade
pip install cachey --upgrade
pip install gputil gpustat --upgrade
pip install arbol --upgrade
pip install simple-zarr-server requests aiohttp --upgrade
pip install python-telegram-bot --upgrade
