========
Install
========

How to install **dexp**
########################

Prerequisites:
**************

**dexp** works on OSX and Windows, but it is recomended to use the latest version of Ubuntu.
We recommend a machine with a top-of-the-line NVIDIA graphics card (min 12G to be confortable).

First, make sure to have a `working python installation <https://github.com/royerlab/dexp/wiki/install_python>`_ .
Second, make sure to have a compatible and functional `CUDA installation <https://github.com/royerlab/dexp/wiki/install_cuda>`_ .

Once these prerequisites are satified, you can install **dexp**.

Installation:
*************

**dexp** can simply be installed with:

To installs **dexp** with GPU support (CUDA 11.2) do:

.. code-block:: bash

   pip install dexp[cuda112]

Other available CUDA versions (from `CuPy <https://cupy.dev/>`_ ) are: cuda111, cuda110, cuda102, cuda101, cuda100.

If instead you do not wish to add CUDA support, you can instead do:

.. code-block:: bash

   pip install dexp


Quick one-line environment setup and installation:
***************************************************

The following line will delete any existing dexp environment, recreate it, and install **dexp** with support for CUDA 11.2:

.. code-block:: bash

   conda deactivate
   conda env remove --name dexp
   conda create -y --name dexp python=3.8
   conda activate dexp
   pip install dexp[cuda112]


Leveraging extra CUDA libraries for faster processing:
*******************************************************

If you want you **dexp** installation to be even faster, you can install additional libraries such as CUDNN and CUTENSOR
with the following command:

.. code-block:: bash

   python setup.py cudalibs --cuda 11.2

Change the version accordingly...
