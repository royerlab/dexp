# Run this to create a new env:
#conda deactivate; conda env remove --name dexp; conda create -y --name dexp python=3.8; conda activate dexp; ./install_linux.sh


conda config --add channels conda-forge
conda install -y -c conda-forge cudatoolkit dask zarr ocl-icd-system pyopencl tifffile scikit-image numba
conda install -y -c pytorch pytorch torchvision torch-dct
#=11.0.221

#pip install cupy-cuda111
pip install cupy-cuda111 -f https://github.com/cupy/cupy/releases/tag/v9.0.0a1

pip install numba --upgrade
pip install torch-dct --upgrade
pip install napari[all] --upgrade
pip install click cachey numexpr joblib --upgrade
pip install spimagine --upgrade
pip install pytest --upgrade

##pip install tensorflow-gpu==1.15 keras==2.2.4 --upgrade


##pip install gputools dtcwt csbdeep --upgrade
#pip install git+https://github.com/guiwitz/naparimovie.git@master#egg=naparimovie --upgrade


#pip install -e .




#pip install scikit-image --upgrade