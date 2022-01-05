# Visualizing your data

DEXP relies on [napari](https://napari.org/) to visualize the data, so make sure it's installed.

Once napari is installed there are two ways you can view your dataset.

- With our DEXP's command line, for example:

    ```bash
    dexp view path_to_your_dataset.zarr --aspect 2.0
    ```

    Execute `dexp view --help` to see additional options.

- Using DEXP's napari plugin:

    1. Install the `napari-dexp` with
        ```bash
        pip install napari-dexp
        ```

    2. Open your dataset with napari as usual
        ```bash
        napari path_to_your_dataset.zarr
        ```

    NOTE: Depending on the ordering of the readers' plugin other plugins might try reading the data first, which could result in errors or warnings.
