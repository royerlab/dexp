========
Install
========

How to install **dexp**
########################

Prerequisites:
**************

**dexp** works on OSX and Windows, but it is recomended to use the latest version of Ubuntu.
We recommend a machine with NVIDIA graphics card with at least 12G.

First, make sure to have a `working python installation <https://github.com/royerlab/dexp/wiki/install_python>`_ .
Second, make sure to have a compatible and functional `CUDA installation <https://github.com/royerlab/dexp/wiki/install_cuda>`_ .

Once these prerequisites are satified, you can install **dexp**.

Installation:
*************

**dexp** can simply be installed with:

To installs **dexp** with GPU support (CUDA 11.2) do:

.. code-block:: bash

   conda install cupy
   pip install dexp[optional,colored]

If instead you do not wish to add CUDA support, you can instead do:

.. code-block:: bash

   pip install dexp[optional,colored]


Quick conda environment setup and installation:
***********************************************

The following commands delete any existing dexp environment, recreate it, install **dexp** with CUDA support and `napari <https://napari.org/>`_:

.. code-block:: bash

   conda deactivate
   conda env remove --name dexp
   conda create -y --name dexp python=3.9
   conda activate dexp
   conda install cupy
   pip install dexp[optional,colored]
   pip install napari[all]


If you are having problems with the cuda/cuda-toolkit the best is to use conda to install the correct version of the cudatoolkit:

.. code-block:: bash

   conda install -c conda-forge cudatoolkit==11.2.2


You can check `here <https://anaconda.org/conda-forge/cudatoolkit/files>`_ for the best matching version.

Notes:
- You might get some error messages recommending you install missing libraries such as CUDNN, CuTensor, nccl, etc... These messages often come with instructions on what to do.
- Adjust your driver version (here 11.2) to your card(s) and drivers.
- Windows users should call :meth:`conda install -c conda-forge pyopencl` before running the second to last step.


Leveraging extra CUDA libraries for faster processing:
******************************************************

If you want you **dexp** CUDA-based processing to be even faster, you can install additional libraries such as CUDNN and CUTENSOR
with the following command:

.. code-block:: bash

   conda install -y -c conda-forge cudnn cutensor nccl

or

.. code-block:: bash

   dexp-install cudalibs 11.2

Change the CUDA version accordingly.
