
conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -y dask zarr
conda install -y tifffile
pip install pybind11 Mako
pip install tensorflow-gpu==1.15 keras
pip install napari click cachey numexpr joblib
pip install ./wheels/pyopencl-2019.1.2+cl12-cp37-cp37m-win_amd64.whl
pip install gputools dtcwt csbdeep cupy-cuda100
pip install -e .

