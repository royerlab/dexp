![fishcolorproj](https://user-images.githubusercontent.com/1870994/113943035-b61b0c00-97b6-11eb-8cfd-ac78e2976ebb.png)
# **dexp** | light-sheet Dataset EXploration and Processing 

**dexp** is a [napari](https://napari.org/), [CuPy](https://cupy.dev/), [Zarr](https://zarr.readthedocs.io/en/stable/), and [DASK](https://dask.org/) based library for managing, processing and visualizing light-sheet microscopy datasets. It consists in light-sheet specialised image processing functions (equalisation, denoising, dehazing, registration, fusion, stabilization, deskewing, deconvolution), visualization functions (napari-based viewing, 2D/3D rendering, video compositing and resizing, mp4 generation), as well as dataset management functions (copy, crop, concatenation, tiff conversion). Almost all functions are GPU accelerated via [CuPy](https://cupy.dev/) but also have a [numpy](https://numpy.org/)-based fallback option for testing on small datasets. In addition to a functional API, DEXP offers a command line interface that makes it very easy for non-coders to pipeline large processing jobs all the way from a large multi-terabyte raw dataset to fully processed and rendered video in MP4 format. 


## How to install **dexp**

### Prerequisites:

**dexp** works on OSX and Windows, but it is recomended to use the latest version of Ubuntu.
We recommend a machine with a top-of-the-line NVIDIA graphics card (min 12G to be confortable).

First, make sure to have a [working python installation](https://github.com/royerlab/dexp/wiki/Installing-Python).  
Second, make sure to have a compatible and functional [CUDA installation](https://github.com/royerlab/dexp/wiki/Installing-CUDA)

Once these prerequisites are satified, you can install **dexp**.

### Installation:

**dexp** can simply be installed with:

To installs **dexp** with GPU support (CUDA 11.2), colored console output, and [napari](https://napari.org/) support do:
```
pip install dexp[color, cuda112, napari]
```
Other available CUDA versions (from [CuPy](https://cupy.dev/)) are: cuda111, cuda110, cuda102, cuda101, cuda100. We recommend using the most recent CUDA version that your system supports, and avoiding versions below 10.0

If instead you do not wish to add CUDA support, you can instead do:
```
pip install dexp
```

**For OSX users:** Before installating dexp, you will first need to install cairo:
```
brew install cairo
```

### Quick one-line environment setup and installation:

The following line will delete any existing dexp environment, recreate it, and install **dexp** with support for CUDA 11.2:
```
conda deactivate; conda env remove --name dexp; conda create -y --name dexp python=3.8; conda activate dexp; pip install dexp[color,cuda112,napari]
```

### Leveraging extra CUDA libraries for faster processing:

If you want you **dexp** CUDA-based processing to be even faster, you can install additional libraries such as CUDNN and CUTENSOR 
with the following command:

```
install cudalibs 11.2
```
Change the CUDA version accordingly...

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
