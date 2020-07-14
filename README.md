
## DEXP
Dataset Exploration Tool


DEXP let's you view both ClearControl and Zarr formats, and copy (convert) from ClearControl format
to Zarr. You can select channels and slice which lets you crop aerbitrarily in time and space.
You can also query information for both formats. 

DEXP should be run ideally on very fast machines very close to the storage,
so we need to think about docker images and such...

Long term, features will include more processing options while copying, such as denoising, fusion,
registration, decopnvolution, etc... 
This is meant to be _the_ tool for processing our data into beautifull timelapses -- excluding analysis.

Once Napari supports 3D viewing we will also have that...

# Prerequisites:

Install Anaconda:
https://www.anaconda.com/distribution/

On windows, make sure to insatll CUDA 10.2 (exactly that version for the current version of dexp)

# Installation:

### Clone dexp:
```
git clone https://github.com/royerlab/dexp.git
```

### Create conda environment:
```
conda create --name dexp python=3.7 
```

### Activate environment:
```
conda activate dexp
```

### Install dependencies:

On a Linux system:
```
bash install_linux.sh
```
or on Windows:
```
bash install_windows.sh
```

If some errors occur -- in paticular related to pypencl r mako -- please rerun the script.

On Ubuntu, you might still have issues, yu might need to install these packages (as root):
```
apt-get install opencl-headers
apt-get install ocl-icd-opencl-dev
apt-get install ocl-icd-libopencl1
```

You might want to install clinf to check your OpenCL install:
```
apt-get install beignet clinfo
```

### Install Aydin:

DEXP depends on Aydin (for denoising, deconvolution and more...)
```
cd aydin
pip install -e .
python setup.py develop
```

### Install dexp:
```
cd dexp
pip install -e .
```

# Usage:

Always make sure that you are in the correct environment:
```
source activate dexp
```

There is currently 5 dexp commands: info, copy, fuse, tiff and view:

## info:
```
Usage: dexp info [OPTIONS] INPUT_PATH

  Retrieves all available information about the dataset.

Options:
  --help  Show this message and exit.

```

## copy:
```
Usage: dexp copy [OPTIONS] INPUT_PATH

  Copies a dataset in ZARR format or CC format. Slicing, projection and
  channel selection are available.

Options:
  -o, --output_path TEXT
  -c, --channels TEXT     list of channels, all channels when ommited.
  -s, --slice TEXT        dataset slice (TZYX), e.g. [0:5] (first five stacks)
                          [:,0:100] (cropping in z)
  -z, --codec TEXT        compression codec: ‘zstd’, ‘blosclz’, ‘lz4’,
                          ‘lz4hc’, ‘zlib’ or ‘snappy’
  -w, --overwrite         to force overwrite of target
  -p, --project INTEGER   max projection over given axis (0->T, 1->Z, 2->Y,
                          3->X)
  --help                  Show this message and exit.

```

## fuse:
```
Usage: dexp fuse [OPTIONS] INPUT_PATH

  Fuses a multi-view dataset.

Options:
  -o, --output_path TEXT
  -s, --slice TEXT        dataset slice (TZYX), e.g. [0:5] (first five stacks)
                          [:,0:100] (cropping in z)
  -z, --codec TEXT        compression codec: ‘zstd’, ‘blosclz’, ‘lz4’,
                          ‘lz4hc’, ‘zlib’ or ‘snappy’
  -w, --overwrite         to force overwrite of target
  -m, --mode [fast]       Available fusion algorithms.
  --help                  Show this message and exit.


```

## tiff:
```
Usage: dexp tiff [OPTIONS] INPUT_PATH

  Exports a dataset to TIFF format.

Options:
  -o, --output_path TEXT
  -c, --channel TEXT      selected channel.
  -s, --slice TEXT        dataset slice (TZYX), e.g. [0:5] (first five stacks)
                          [:,0:100] (cropping in z)
  -w, --overwrite         to force overwrite of target
  --help                  Show this message and exit.

```

## view:
```
Usage: dexp view [OPTIONS] INPUT_PATH

  Opens dataset for viewing using napari.

Options:
  -c, --channels TEXT  list of channels, all channels when ommited.
  -s, --slice TEXT     dataset slice (TZYX), e.g. [0:5] (first five stacks)
                       [:,0:100] (cropping in z).
  -v, --volume         to view with volume rendering (3D ray casting)
  --help               Show this message and exit.
```


# Examples:

The folowing examples can be tested on existing datasets:

Gathers info on the dataset '2019-07-03-16-23-15-65-f2va.mch7' with napari: 
```
dexp info 2019-07-03-16-23-15-65-f2va.mch7
```

Copies a z-max-projection for the first 10n timepoints (-s [0:10]) from scope acquisition '2019-07-03-16-23-15-65-f2va.mch7' to Zarr folder '~/Downloads/local.zarr', and overwrites whatever is there (-w). 
```
dexp copy -w -p 1 -s '[0:10]' 2019-07-03-16-23-15-65-f2va.mch7 -o ~/Downloads/local.zarr
```

Views the first 100 times points '2019-07-03-16-23-15-65-f2va.mch7' with napari: 
```
dexp view -s '[0:100]' 2019-07-03-16-23-15-65-f2va.mch7
```

# Technical notes:

- You can pass arguments in any order for bash but not for all shells, i.e. zsh.
- You can pass string arguments as non string in bash but not in all shells.

