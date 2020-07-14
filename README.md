
## DEXP
Dataset Exploration Tool


DEXP let's you view both ClearControl and Zarr formats, and copy (convert) from ClearControl format
to Zarr. You can select channels and slice which lets you crop arbitrarily in time and space.
You can also query information for both formats. 

DEXP ZARR storage supports both directory and zip storage and different compression codecs. Expect a typicall factor 3
compression for raw data, 10x compression for deconvolved and/or denoised data, and up to 90x compression for sparse expression.
Substracting the backgroundlight level (~ around 100 for sCMOS cameras) brings compression from typically 3X on raw data to almost 10x.

DEXP currently supports fusion&registration, deconvolution, isonet, viweing with napari, volumetric rendering,
video compositing (merging channels), video export to mp4, export to tiff. 

DEXP should be run ideally on very fast machines very close to the storage.



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

There is currently 12dexp commands:   

Sytorage info & manipulation commands:
  info
  copy
  add
  
Processing commands:
  fuse
  deconv
  isonet
  
Video rendering commands:
  render
  blend
  stack
  mp4
 
Export commnads:
  tiff
  
Viewing commands:
  view

## info:
Collects information about a given dataset:
```
Usage: dexp info [OPTIONS] INPUT_PATH

Options:
  --help  Show this message and exit.

```

## copy:
Copies a dataset from one file/folder to another file/folder.
Teh destination is _always_ in ZARR format (dir ort zip).
Prefer 'zip' for fully processed datasets (<200GB) to be able to 
convenienytly copy a single file instead of a gazillion failes.
```
Usage: dexp copy [OPTIONS] INPUT_PATH

Options:
  -o, --output_path TEXT
  -c, --channels TEXT     List of channels, all channels when ommited.
  -s, --slicing TEXT      Dataset slice (TZYX), e.g. [0:5] (first five stacks)
                          [:,0:100] (cropping in z)

  -st, --store TEXT       Store: ‘dir’, ‘zip’  [default: dir]
  -z, --codec TEXT        Compression codec: zstd for ’, ‘blosclz’, ‘lz4’,
                          ‘lz4hc’, ‘zlib’ or ‘snappy’   [default: zstd]

  -l, --clevel INTEGER    Compression level  [default: 3]
  -w, --overwrite         Forces overwrite of target  [default: False]
  -p, --project INTEGER   max projection over given axis (0->T, 1->Z, 2->Y,
                          3->X)

  -k, --workers INTEGER   Number of worker threads to spawn.  [default: 1]
  --help                  Show this message and exit.

```

## add:
Adds a channel from one dataset to another (possibly not yet existant) ZARR file/folder.
Channels can be renamed as they are copied.
```
Usage: dexp add [OPTIONS] INPUT_PATH

Options:
  -o, --output_path TEXT
  -c, --channels TEXT     List of channels, all channels when ommited.
  -rc, --rename TEXT      You can rename channels: e.g. if channels are
                          `channel1,anotherc` then `gfp,rfp` would rename the
                          `channel1` channel to `gfp`, and `anotherc` to `rfp`

  -st, --store TEXT       Store: ‘dir’, ‘zip’  [default: dir]
  -z, --codec TEXT        Compression codec: zstd for ’, ‘blosclz’, ‘lz4’,
                          ‘lz4hc’, ‘zlib’ or ‘snappy’   [default: zstd]

  -l, --clevel INTEGER    Compression level  [default: 3]
  -w, --overwrite         Forces overwrite of target  [default: False]
  --help                  Show this message and exit.
```

## Fusion (& registration):
```
Usage: dexp fuse [OPTIONS] INPUT_PATH

Options:
  -o, --output_path TEXT
  -s, --slicing TEXT        dataset slice (TZYX), e.g. [0:5] (first five
                            stacks) [:,0:100] (cropping in z)

  -st, --store TEXT         Store: ‘dir’, ‘zip’  [default: dir]
  -z, --codec TEXT          compression codec: ‘zstd’, ‘blosclz’, ‘lz4’,
                            ‘lz4hc’, ‘zlib’ or ‘snappy’

  -l, --clevel INTEGER      Compression level  [default: 3]
  -w, --overwrite           to force overwrite of target  [default: False]
  -k, --workers INTEGER     Number of worker threads to spawn, recommended: 1
                            (unless you know what you are doing)  [default: 1]

  -zl, --zerolevel INTEGER  Sets the 'zero-level' i.e. the pixel values in the
                            background (to be substracted)  [default: 110]

  -ls, --loadshifts         Turn on to load the registration parameters (i.e
                            translation shifts) from another run  [default:
                            False]

  --help                    Show this message and exit.
```

## Deconvolution:
```
Usage: dexp deconv [OPTIONS] INPUT_PATH

Options:
  -o, --output_path TEXT
  -c, --channels TEXT           list of channels, all channels when ommited.
  -s, --slicing TEXT            dataset slice (TZYX), e.g. [0:5] (first five
                                stacks) [:,0:100] (cropping in z)

  -st, --store TEXT             Store: ‘dir’, ‘zip’  [default: dir]
  -z, --codec TEXT              compression codec: ‘zstd’, ‘blosclz’, ‘lz4’,
                                ‘lz4hc’, ‘zlib’ or ‘snappy’   [default: zstd]

  -l, --clevel INTEGER          Compression level  [default: 3]
  -w, --overwrite               to force overwrite of target  [default: False]
  -k, --workers INTEGER         Number of worker threads to spawn,
                                recommended: 1 (unless you know what you are
                                doing)  [default: 1]

  -m, --method TEXT             Deconvolution method: for now only lr (Lucy
                                Richardson)  [default: lr]

  -i, --iterations INTEGER      Number of deconvolution iterations. More
                                iterations takes longer, will be sharper, but
                                might also be potentially more noisy depending
                                on method.  [default: 15]

  -mc, --maxcorrection INTEGER  Max correction in folds per iteration. Noisy
                                datasets benefit from mc=2 (recommended), for
                                noiseless datasets you can push to mc=8 or
                                even more.  [default: 2]

  -pw, --power FLOAT            Correction exponent, default for standard LR
                                is 1, set to 1.5 for acceleration.  [default:
                                1.5]

  -dxy, --dxy FLOAT             Voxel size along x and y in microns  [default:
                                0.485]

  -dz, --dz FLOAT               Voxel size along z in microns  [default: 1.94]
  -sxy, --xysize INTEGER        Voxel size along xy in microns  [default: 17]
  -sz, --zsize INTEGER          Voxel size along z in microns  [default: 31]
  -d, --downscalexy2 TEXT       Downscales along x and y for faster
                                deconvolution (but worse quality of course)

  --help                    Show this message and exit.
```

## Isonet:
```
Usage: dexp isonet [OPTIONS] INPUT_PATH

Options:
  -o, --output_path TEXT
  -s, --slicing TEXT        dataset slice (TZYX), e.g. [0:5] (first five
                            stacks) [:,0:100] (cropping in z)

  -st, --store TEXT         Store: ‘dir’, ‘zip’  [default: dir]
  -z, --codec TEXT          compression codec: ‘zstd’, ‘blosclz’, ‘lz4’,
                            ‘lz4hc’, ‘zlib’ or ‘snappy’   [default: zstd]

  -l, --clevel INTEGER      Compression level  [default: 3]
  -w, --overwrite           to force overwrite of target  [default: False]
  -c, --context TEXT        IsoNet context name  [default: default]
  -m, --mode TEXT           mode: 'pta' -> prepare, train and apply, 'a' just
                            apply    [default: pta]

  -e, --max_epochs INTEGER  to force overwrite of target  [default: 50]
  --help                    Show this message and exit.

```


## Volume rendering:
```
Usage: dexp render [OPTIONS] INPUT_PATH

Options:
  -o, --output_path TEXT     Output folder to store rendered PNGs. Default is:
                             frames_<channel_name>

  -c, --channels TEXT        list of channels to render, all channels when
                             ommited.

  -s, --slicing TEXT         dataset slice (TZYX), e.g. [0:5] (first five
                             stacks) [:,0:100] (cropping in z).

  -w, --overwrite            to force overwrite of target  [default: False]
  -a, --aspect FLOAT         sets aspect ratio e.g. 4  [default: 4]
  -cm, --colormap TEXT       sets colormap, e.g. viridis, gray, magma, plasma,
                             inferno   [default: magma]

  -rs, --rendersize INTEGER  Sets the frame render size. i.e. -ws 400 sets the
                             window to 400x400  [default: 2048]

  -cl, --clim TEXT           Sets the contrast limits, i.e. -cl 0,1000 sets
                             the contrast limits to [0,1000]

  -opt, --options TEXT       Render options, e.g: 'gamma=1.2,box=True'.
                             Important: no white spaces!!! Complete list with
                             defaults will be displayed on first run

  --help                     Show this message and exit.

```

## Video compositing:
```
Usage: dexp blend [OPTIONS] [INPUT_PATHS]...

Options:
  -o, --output_path TEXT  Output folder for blended frames.
  -b, --blending TEXT     Blending mode: max, add, addclip, adderf (add stands
                          for addclip).  [default: max]

  -w, --overwrite         to force overwrite of target  [default: False]
  -k, --workers INTEGER   Number of worker threads to spawn, set to -1 for
                          maximum number of workers  [default: -1]

  --help                  Show this message and exit.

```

## Video compositing:
```
Usage: dexp blend [OPTIONS] [INPUT_PATHS]...

Options:
  -o, --output_path TEXT  Output folder for blended frames.
  -b, --blending TEXT     Blending mode: max, add, addclip, adderf (add stands
                          for addclip).  [default: max]

  -w, --overwrite         to force overwrite of target  [default: False]
  -k, --workers INTEGER   Number of worker threads to spawn, set to -1 for
                          maximum number of workers  [default: -1]

  --help                  Show this message and exit.

```


## Video stacking -- horyzontal or vertical:
```
Usage: dexp stack [OPTIONS] [INPUT_PATHS]...

Options:
  -o, --output_path TEXT  Output folder for blended frames.
  -r, --orientation TEXT  Stitching mode: horiz, vert  [default: horiz]
  -w, --overwrite         to force overwrite of target  [default: False]
  -k, --workers INTEGER   Number of worker threads to spawn, set to -1 for
                          maximum number of workers  [default: -1]

  --help                  Show this message and exit.

```

## Conversion from frame sequences to mp4 file:
```
Usage: dexp mp4 [OPTIONS] INPUT_PATH

Options:
  -o, --output_path TEXT     Output file path for movie
  -fps, --framerate INTEGER  Sets the framerate in frames per second
                             [default: 30]

  -w, --overwrite            to force overwrite of target  [default: False]
  --help                     Show this message and exit.

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

