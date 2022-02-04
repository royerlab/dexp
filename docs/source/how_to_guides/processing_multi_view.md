
# Processing a multi-view dataset using DEXP CLI

In this example, we will show how to process the multi views acquired by our multi-view light-sheet microscope [[1]](#1) to
obtain a single fused image.

The processing consists of two main steps:
- Fusion: to fuse the multiple views into a single image.
- Deconvolution: to reduce the illumination spread from the acquisition process. Ideally, we would know the true point spread function (PSF), usually we don't have access to this and use the theoretical PSF calculated from the imaging system parameters.

One could also invert the processing steps by deconvolving each view separately and then fusing, while this could improve the final result the processing time will be much longer.

To reproduce this tutorial you can [download a sample](https://drive.google.com/file/d/1W9ysPcup6iW7E1CzTL2c1oERwBn8pj0D/view?usp=sharing) of a single time point with 4 views from the combination of two cameras (C0 and C1) and two light sheets (L0 and L1) of a Zebra-fish embryo. This dataset will be referred as `demo_4views.zarr.zip`.

NOTE: We recommend to avoid using the `.zarr.zip` format and use `.zarr` because it can lead to **data corruption** if a process is interrupted without closing the opened file. Here we used the `.zarr.zip` to share the file more easily and the remaining data is stored with only `.zarr`.

## Inspecting the dataset

We can inspect the dataset through the command line using `dexp info`

```bash
$ dexp info demo_4views.zarr.zip
```

This dumps a bunch of information, one of them being the zarr structure tree, which shows that our dataset contains 4 different channels named `['488-C0L0', '488-C0L1', '488-C1L0', '488-C1L1']`, each one containing an image with shape `(1, 407, 2048, 2048)` representing the time, z, y, and x axes, and their marginal projections (e.g., `488-C0L0_projection_0`).

```
...
├╗ Information on dataset at: ['demo_4views.zarr.zip']
│├ Dataset at location: demo_4views.zarr.zip
││ Channels: ['488-C0L0', '488-C0L1', '488-C1L0', '488-C1L1']
││ Zarr tree:
││ /
││  ├── 488-C0L0
││  │   ├── 488-C0L0 (1, 407, 2048, 2048) uint16
││  │   ├── 488-C0L0_projection_0 (1, 2048, 2048) uint16
││  │   ├── 488-C0L0_projection_1 (1, 407, 2048) uint16
││  │   └── 488-C0L0_projection_2 (1, 407, 2048) uint16
││  ├── 488-C0L1
││  │   ├── 488-C0L1 (1, 407, 2048, 2048) uint16
││  │   ├── 488-C0L1_projection_0 (1, 2048, 2048) uint16
││  │   ├── 488-C0L1_projection_1 (1, 407, 2048) uint16
││  │   └── 488-C0L1_projection_2 (1, 407, 2048) uint16
││  ├── 488-C1L0
││  │   ├── 488-C1L0 (1, 407, 2048, 2048) uint16
││  │   ├── 488-C1L0_projection_0 (1, 2048, 2048) uint16
││  │   ├── 488-C1L0_projection_1 (1, 407, 2048) uint16
││  │   └── 488-C1L0_projection_2 (1, 407, 2048) uint16
││  └── 488-C1L1
││      ├── 488-C1L1 (1, 407, 2048, 2048) uint16
││      ├── 488-C1L1_projection_0 (1, 2048, 2048) uint16
││      ├── 488-C1L1_projection_1 (1, 407, 2048) uint16
││      └── 488-C1L1_projection_2 (1, 407, 2048) uint16.
...
```

Additionally, we can check the command line history, that stores the command used to generate this dataset. In this case, a dataset was created in `napari` when viewing an image named `embryo_4views.tif`, and later this data was copied from `.zarr` to `.zarr.zip`.

```
...
││ Command line history:
││  ├──■ 'napari embryo_4views.tif'
││  └──■ 'dexp copy demo_4views.zarr -o demo_4views.zarr.zip'
...
```

## Fusion

To fuse a dataset we must first register the views from different cameras. We assume the displacement (i.e., translation) between cameras is constant through out the multiple frames (considering the general case, here we have a single time point). Therefore, we summarize the registration parameters as the median displacement between all time points. This parameters are saved in the `registration_models.txt` by default.

The `-c` flag indicates the channels we are using to register the views, this is not necessary when all channels should be considered, which is not always the case.

```bash
$ dexp register demo_4views.zarr.zip -c 488-C0L0,488-C0L1,488-C1L0,488-C1L1
```

With this we can fuse the views, the options `--pad` indicates the data should be padded when aligning the views, and `--loadreg` that it should load the registration models and not compute it during processing.

```bash
$ dexp fuse demo_4views.zarr.zip --pad --loadreg -c 488-C0L0,488-C0L1,488-C1L0,488-C1L1
```

By default the output is saved with an additional `_fused` to its name. Hence, we can visualize its results using (package `napari-dexp` is required.)

```bash
$ napari demo_4views_fused.zarr
```

Notice that the fused dataset contains only a single image.

```
$ dexp info demo_4views_fused.zarr
...
├╗ Information on dataset at: ['demo_4views_fused.zarr']
│├ Dataset at location: demo_4views_fused.zarr
││ Channels: ['fused']
││ Zarr tree:
││ /
││  └── fused
││      ├── fused (1, 408, 2048, 2078) uint16
││      ├── fused_projection_0 (1, 2048, 2078) uint16
││      ├── fused_projection_1 (1, 408, 2078) uint16
││      └── fused_projection_2 (1, 408, 2048) uint16.
...
```

## Deconvolution

For the deconvolution you need to provide information of your microscopes PSF. The PSFs from our microscopes are already available through the `--objective (-obj)` flag, you can change their configuration with the `dxy, dz, na` of your acquisition system, or open a github issue to discuss the addition of other PSFs. We also use the option `--max-correction (-mc)` to limit the multiplicative correction of lucy-richardson deconvolution to avoid numerical errors caused by the very large values from deconvolving the beads present in this dataset --- this is not necessary most of the time.

```bash
$ dexp deconv demo_4views_fused.zarr -obj nikon16x08na -mc 5
```

Finally, the data processing is done and the final dataset can be visualized with:

```bash
$ napari demo_4views_fused_deconv.zarr
```

And the command line history reports what operations were used to produce it:

```
$ dexp info demo_4views_fused_deconv.zarr
...
││ Command line history:
││  ├──■ 'napari embryo_4views.tif'
││  ├──■ 'dexp copy demo_4views.zarr -o demo_4views.zarr.zip'
││  ├──■ 'dexp fuse demo_4views.zarr.zip --pad --loadreg -c 488-C0L0,488-C0L1 488-C1L0,488-C1L1'
││  └──■ 'dexp deconv demo_4views_fused.zarr -obj nikon16x08na -mc 5
...
```

We encourage you to play with the other parameters to obtain the optimal results. Their description can be found through `dexp --help` or `dexp <command (e.g. deconv)> --help`.

## References

<a id="1">[1]</a>
Royer, Loïc A., et al.
"Adaptive light-sheet microscopy for long-term, high-resolution imaging in living organisms."
Nature biotechnology 34.12 (2016): 1267-1278.
