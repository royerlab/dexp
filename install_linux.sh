
conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -y dask zarr
conda install -y tifffile
pip install cupy-cuda101
pip install torch-dct
pip install scikit-image
pip install pybind11 Mako
pip install tensorflow-gpu==1.15 keras==2.2.4
pip install napari[all] --upgrade
pip install click cachey numexpr joblib
pip install pyopencl
pip install gputools dtcwt csbdeep cupy-cuda100
pip install git+https://github.com/guiwitz/naparimovie.git@master#egg=naparimovie
pip install spimagine
pip install -e .

