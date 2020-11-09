# Run this to create a new env:
#conda deactivate; conda env remove --name dexp; conda create -y --name dexp python=3.8; conda activate dexp; ./install_osx.sh


conda config --add channels conda-forge
conda install -y -c ocl-icd-system
conda install -y -c conda-forge dask dask-image zarr  pyopencl tifffile scikit-image numba numcodecs
conda install -y -c conda-forge cudatoolkit
conda install -y -c pytorch pytorch torchvision torch-dct


#pip install cupy-cuda111
pip install cupy-cuda111 -f https://github.com/cupy/cupy/releases/tag/v9.0.0a1

pip install scikit-image --upgrade
pip install numba --upgrade
pip install torch-dct --upgrade
pip install napari[all] --upgrade
pip install click cachey numexpr joblib --upgrade
pip install spimagine --upgrade
pip install pytest --upgrade

pip install -e .
