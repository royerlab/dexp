# Run this to create a new env:
#conda deactivate; conda env remove --name dexp; conda create -y --name dexp python=3.8; conda activate dexp; ./install_linux.sh

pip install --upgrade pip

pip install numpy --upgrade
pip install scipy --upgrade
pip install numba --upgrade
pip install numcodecs --upgrade
pip install scikit-image --upgrade
pip install tifffile --upgrade
pip install numexpr --upgrade
pip install joblib --upgrade
pip install pytest --upgrade
pip install graphviz --upgrade
pip install click --upgrade
pip install dask --upgrade
pip install dask-image --upgrade
pip install distributed --upgrade
pip install zarr --upgrade

pip install napari[all] --upgrade
pip install cachey --upgrade
pip install spimagine --upgrade
pip install gputil gpustat --upgrade
pip install arbol colorama ansicolors --upgrade
pip install simple-zarr-server requests aiohttp --upgrade
pip install colorcet --upgrade
pip install python-telegram-bot --upgrade
pip install PyYAML --upgrade

sudo apt install libcairo2-dev pkg-config python3-dev
pip install pycairo==1.11.1

pip install cupy-cuda111==9.0.0b2 --no-cache-dir

python -m cupyx.tools.install_library --library cudnn --cuda 11.1
python -m cupyx.tools.install_library --library cutensor --cuda 11.1

pip install -e .
