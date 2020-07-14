
conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -y dask zarr 
conda install -y tifffile
conda update  dask zarr tifffile pytorch
pip install cupy-cuda102 --upgrade
pip install torch-dct --upgrade
pip install scikit-image --upgrade
pip install pybind11 Mako --upgrade
pip install tensorflow-gpu==1.15 keras==2.2.4 --upgrade
pip install napari[all] --upgrade
pip install click cachey numexpr joblib --upgrade
pip install pyopencl --upgrade
pip install gputools dtcwt csbdeep --upgrade
pip install git+https://github.com/guiwitz/naparimovie.git@master#egg=naparimovie --upgrade
pip install spimagine --upgrade
pip install -e .

