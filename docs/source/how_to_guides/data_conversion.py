from tqdm import tqdm
import numpy as np

from dexp.datasets import ZDataset


SHAPE = (512, 1024, 1024)


def load_time_point(time_point: int, channel: str) -> np.ndarray:
    """Dummy function to simulate data loading from another file format.
    """
    # IMPORTANT: here you should read your data
    return np.random.randint(0, 1500, size=SHAPE, dtype=np.uint16)


if __name__ == "__main__":

    ds = ZDataset('new_dataset.zarr', mode='w-')

    # shape and datatype should be know when creating the dataset
    gfp_shape = (25, ) + SHAPE   # (25, 512, 1024, 1024)

    ds.add_channel(name='GFP', shape=gfp_shape, dtype=np.uint16)

    for tp in tqdm(range(ds.nb_timepoints('GFP')), 'Converting GFP channel'):
        stack = load_time_point(tp, 'GFP')  # loading the data
        ds.write_stack('GFP', tp, stack)  # saving to our format

    # the channels can have different shapes
    rfp_shape = (10, ) + SHAPE

    ds.add_channel(name='RFP', shape=gfp_shape, dtype=np.uint16)

    for tp in tqdm(range(ds.nb_timepoints('RFP')), 'Converting RFP channel'):
        stack = load_time_point(tp, 'RFP')
        ds.write_stack('RFP', tp, stack)
    
    # done
    ds.close()
