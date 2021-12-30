from typing import Dict, Sequence

import numpy as np
from ome_zarr.format import CurrentFormat


def default_omero_metadata(name: str, channels: Sequence[str], dtype: np.dtype) -> Dict:
    val_max = 1.0 if np.issubdtype(dtype, np.floating) else np.iinfo(dtype).max

    return {
        "id": 1,
        "name": name,
        "version": CurrentFormat().version,
        "channels": [
            {
                "active": True,
                "coefficient": 1,
                "color": "FFFFFF",
                "family": "linear",
                "inverted": False,
                "label": ch,
                "window": {"start": 0, "end": 1500, "min": 0, "max": val_max},  # guessing
            }
            for ch in channels
        ],
    }
