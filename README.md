![fishcolorproj](https://user-images.githubusercontent.com/1870994/113943035-b61b0c00-97b6-11eb-8cfd-ac78e2976ebb.png)
# **dexp** | light-sheet Dataset EXploration and Processing

**dexp** is a [napari](https://napari.org/), [CuPy](https://cupy.dev/), [Zarr](https://zarr.readthedocs.io/en/stable/), and [DASK](https://dask.org/) based library for managing, processing and visualizing light-sheet microscopy datasets. It consists in light-sheet specialised image processing functions (equalisation, denoising, dehazing, registration, fusion, stabilization, deskewing, deconvolution), visualization functions (napari-based viewing, 2D/3D rendering, video compositing and resizing, mp4 generation), as well as dataset management functions (copy, crop, concatenation, tiff conversion). Almost all functions are GPU accelerated via [CuPy](https://cupy.dev/) but also have a [numpy](https://numpy.org/)-based fallback option for testing on small datasets. In addition to a functional API, DEXP offers a command line interface that makes it very easy for non-coders to pipeline large processing jobs all the way from a large multi-terabyte raw dataset to fully processed and rendered video in MP4 format.


## How to install **dexp**

### Prerequisites:

**dexp** works on OSX and Windows, but it is recomended to use the latest version of Ubuntu.
We recommend a machine with NVIDIA graphics card with at least 12G.

First, make sure to have a [working python installation](https://github.com/royerlab/dexp/wiki/Installing-Python).
Second, make sure to have a compatible and functional [CUDA installation](https://github.com/royerlab/dexp/wiki/Installing-CUDA)

Once these prerequisites are satified, you can install **dexp**.

### Installation:

**dexp** can simply be installed with:

```
pip install dexp
```

To installs **dexp** with GPU support, the optional features and colored console output:
```
conda install cupy
pip install dexp[optional,colored]
```

**For OSX users:** Before installating dexp, you will first need to install cairo:
```
brew install cairo
```

### Quick conda environment setup and installation:

There are multiple options when using a conda environment:

- You can create our suggested DEXP (and some additional packages) environment by, it requires GPU:

   ```
   conda env create --name dexp --file env-linux-gpu.yaml
   ```

- Or create your own conda environment from scratch with the following commands delete any existing dexp environment, recreate it, install **dexp** with CUDA support and [napari](https://napari.org/):
   ```
   conda deactivate
   conda env remove --name dexp
   conda create -y --name dexp python=3.9
   conda activate dexp
   conda install cupy
   pip install dexp[optional,colored]
   pip install napari[all]
   ```

   If you are having problems with the cuda/cuda-toolkit the best is to use conda to install the correct version of the cudatoolkit:
   ```
   conda install -c conda-forge cudatoolkit==11.2.2
   ```
   You can check [here](https://anaconda.org/conda-forge/cudatoolkit/files) for the best matching version.

   Notes:
   - You might get some error messages recommending you install missing libraries such as CUDNN, CuTensor, nccl, etc... These messages often come with instructions on what to do.
   - Adjust your driver version (here 11.2) to your card(s) and drivers.
   - Windows users should call `conda install -c conda-forge pyopencl` before running the second to last step.

### Leveraging extra CUDA libraries for faster processing:

If you want you **dexp** CUDA-based processing to be even faster, you can install additional libraries such as CUDNN and CUTENSOR
with the following command:

```
conda install -y -c conda-forge cudnn cutensor nccl
```

or

```
dexp-install cudalibs 11.2
```

Change the CUDA version accordingly.

### **dexp** Zarr dataset structure

The zarr datasets injested and written by **dexp** are organized as below:

```bash
/ (root)
 └── channel1 (group)
     ├── channel1 (array)
     ├── channel1_projection_0 (optional)
     ├── channel1_projection_1 (optional)
     └── channel1_projection_2 (optional)
  └── channel2 (group)
     ├── channel2 (array)
     ├── channel2_projection_0 (optional)
     ├── channel2_projection_1 (optional)
     └── channel2_projection_2 (optional)
  └── more channels ...
```

Channels (zarr group) could be of a particular emission color (e.g. DAPI, GFP, etc), or/and of a particular imaging views
(e.g. view1 and view2 in a dual view acquisition).
Under each channel group, there could be multiple zarr array. The array that has the same name as the group name is typically
a n-dimentional stack (e.g. time-z-y-x). The projections of that nd array are optional (useful for quick exploration of the
nd stack). When writting output datasets **dexp** automatically generates these projections. Future versions of **dexp** might
add more such convenience arrays, high in the list is of course downscaled version sof the stacks for faster visualisation and
browsing...

Note: Our goal is to eventually transition to a ome-zarr and/or ngff storage by defaut for both reading and writting.
For reading we have also support for specific dataset produced by our light-sheet microscopes, see [here](https://github.com/royerlab/dexp/wiki/dexp-dataset-formats) for supported microscopes and formats. This is currently limited but contributions are very welcome!


### DaXi
DEXP supports processing datasets acquired on the [DaXi microscope](https://github.com/royerlab/daxi) ([paper](https://www.biorxiv.org/content/10.1101/2020.09.22.309229v2)).
You can test processing of DaXi data using an [example dataset](https://drive.google.com/drive/folders/1c-xtJd4INtTll1s1HEbs1rIRF2M7Hg1X)


### Versions

The list of released versions can be found [here](https://pypi.org/project/dexp/#history). The version format is: YYYY.MM.DD.M where YYYY is the year, MM the month, dd the day, and M is the number of elapsed minutes of the day. Git tags are automatically set to link pipy versions to github tagged versions so that the corresponding code can be inspected on github, probably the most important feature. This is a very simple and semantically clear versionning scheme that accomodates for a rapid rate of updates.

### How to use **dexp** ?

In depth documentation can be found [here](https://royerlab.github.io/dexp/index.html) for both command line  commands and for the python API.

### Contributors:

Jordao Bragantini, Ahmet Can Solak, Bin Yang, and Loic A Royer
