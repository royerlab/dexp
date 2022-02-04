# Implementing your own processing pipeline

Here we provide an example where we implement a custom image processing routine and apply it to an DEXP's dataset.

To replicate this example you need to download our sample data [here](https://drive.google.com/file/d/1W9ysPcup6iW7E1CzTL2c1oERwBn8pj0D/view?usp=sharing),
it contains four different views of a zebra-fish embryo. In the [previous tutorial](processing_multi_view.md) we discuss it in more detail.

First, we import the required packages:

.. literalinclude:: implementing_processing.py
  :lines: 1-7

Next we implement our image processing operation to remove the background fluorescence, in this case it consists of 2 steps:

* Filter out noisy spots using morphological opening operator with a small sphere --- using GPU acceleration through `cupy`;

* Subtract darker components with area larger than `10_000` using the area white top hat transform, we use downsampling of `4` to speedup the computation and the axis-0 slices are processed individually due to the data anisotropy.

.. literalinclude:: implementing_processing.py
  :lines: 10-19


To apply this operation to the `demo_4views.zarr.zip` dataset we open it and create a new storage `processed.zarr` using write mode `w-`.

We iterate over each existing channel, creating it respective processed channel in the output dataset, and iterating over their time points. At each iteration, the required stack is read from the input dataset, the processing function is called, and the processed data is written to the output dataset, thus avoiding accumulation of data in the computer memory.

.. literalinclude:: implementing_processing.py
  :lines: 22-

The datasets are closed before finishing the program.

The whole script is presented below:

.. literalinclude:: implementing_processing.py
