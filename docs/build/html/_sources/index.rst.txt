=====
DEXP
=====

.. image:: https://user-images.githubusercontent.com/1870994/113943035-b61b0c00-97b6-11eb-8cfd-ac78e2976ebb.png
    :alt: **dexp** | Light-sheet Dataset EXploration and Processing


Welcome to **dexp** (* Light-sheet Dataset Exploration and Processing*) documentation.

How to use **dexp** ?
**********************

First you need a dataset acquired on a light-sheet microscope, see `here <https://github.com/royerlab/dexp/wiki/dexp_datasets>`_ for supported microscopes and formats.

Second, you can use any of the commands `here <https://github.com/royerlab/dexp/wiki/dexp_commands>`_ to process your data.
The list of commands can be found by :

.. code-block:: bash

   dexp --help


Example usage
*************

TBD.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   Install <getting_started/install.rst>
   Contact us <getting_started/contact_us.rst>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Dexp Commands

   add <dexp_commands/add.rst>
   check <dexp_commands/check>
   copy <dexp_commands/copy>
   deconv <dexp_commands/deconv>
   deskew <dexp_commands/deskew>
   fuse <dexp_commands/fuse>
   info <dexp_commands/info>
   isonet <dexp_commands/isonet>
   projrender <dexp_commands/projrender>
   serve <dexp_commands/serve>
   speedtest <dexp_commands/speedtest>
   stabilize <dexp_commands/stabilize>
   tiff <dexp_commands/tiff>
   view <dexp_commands/view>
   volrender <dexp_commands/volrender>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   Introduction <api/introduction.rst>
   Processing <api/processing>
   Volume Render <api/volume_render>
