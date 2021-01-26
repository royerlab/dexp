# Run this to create a new env:
#conda deactivate; conda env remove --name dexp; conda create -y --name dexp python=3.8; conda activate dexp; ./install_linux.sh

# To install cuTensor, type:
#sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
#sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
#sudo apt update
#sudo apt -y install libcutensor1 libcutensor-dev libcutensor-doc

conda config --add channels anaconda
conda install -y -c anaconda numpy #"libblas=*=*mkl"
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
#conda install -y -c anaconda cudatoolkit
#conda install -y -c rapidsai dask-cuda
#conda install -y -c conda-forge ocl-icd-system
conda install pytorch torchvision torchaudio cudatoolkit=11.0 torch-dct -c pytorch

#pip install cupy-cuda111
#pip install cupy-cuda111 -f https://github.com/cupy/cupy/releases/tag/v9.0.0a1
pip install cupy-cuda111==9.0.0a2

pip install pyopencl --upgrade
#pip install torch-dct --upgrade
pip install napari[all] --upgrade
pip install cachey --upgrade
pip install spimagine --upgrade
pip install gputil --upgrade
pip install arbol colorama ansicolors --upgrade
pip install simple-zarr-server requests aiohttp --upgrade
pip install colorcet --upgrade
pip install python-telegram-bot --upgrade
pip install PyYAML --upgrade

pip install -e .
