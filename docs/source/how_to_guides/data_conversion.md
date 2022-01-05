# Converting to our file format

Our preferred file format is `.zarr` which is accessed in DEXP through `ZDataset` class from `dexp.datasets`.

As long as you can load your dataset as a numpy array you can convert your data into a `ZDataset` as below:

.. literalinclude:: data_conversion.py

The next guide shows how to visualize your data.
