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

# Installation:

Create conda environment:
```
conda create --name dexp python=3.7 
```

Activate environment:
```
conda activate dexp
```

Install important packages:
```
conda install numpy mkl zarr dask click numcodecs joblib
pip install spimagine
```

Clone dexp:
```
git clone https://github.com/royerlab/dexp.git
```

Install dexp:
```
cd dexp
pip install -e .
```

# Usage:

Always make sure that you are in the correct environment:
```
source activate dexp
```

There is currently 3 dexp commands: copy, info and view:

## info:
```
Usage: dexp info [OPTIONS] INPUT_PATH

Options:
  --help  Show this message and exit.
```

## copy:
```
Usage: dexp copy [OPTIONS] INPUT_PATH

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

## view:
```
Usage: dexp view [OPTIONS] INPUT_PATH

Options:
  -c, --channels TEXT  list of channels, all channels when ommited.
  -s, --slice TEXT     dataset slice (TZYX), e.g. [0:5] (first five stacks)
                       [:,0:100] (cropping in z).
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
