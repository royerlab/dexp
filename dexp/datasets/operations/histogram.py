from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from tqdm import tqdm

from dexp.datasets import BaseDataset
from dexp.utils.backends import BestBackend


def dataset_histogram(
    dataset: BaseDataset,
    output_dir: str,
    channels: Sequence[str],
    device: int,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for channel in channels:

        with PdfPages(output_dir / f"histograms_{channel}.pdf") as pdf:

            histograms = []

            data_dir = output_dir / f"{channel}_data"
            data_dir.mkdir(exist_ok=True)

            for i, stack in tqdm(enumerate(dataset[channel]), f"Computing Histogram of {channel}"):

                with BestBackend(device_id=device) as bkd:
                    stack = bkd.to_backend(stack)
                    hist = np.bincount(stack.flatten())
                    hist = bkd.to_numpy(hist)

                np.save(data_dir / f"{channel}_{i}.npy", hist)
                histograms.append(hist)

            max_length = max(len(h) for h in histograms)
            hist_2d = np.zeros((len(histograms), max_length), dtype=int)

            for i, hist in enumerate(histograms):
                hist_2d[i, : len(hist)] += hist

                plt.figure(figsize=(5, 5))
                plt.bar(np.arange(len(hist)), hist, width=1)
                plt.title(f"Histogram of {channel} and index {i}.")
                plt.ylabel("Count")
                plt.yscale("log")
                plt.xlim((0, max_length))
                plt.xlabel("Image intensity")
                pdf.savefig()
                plt.close()

            plot = plt.imshow(hist_2d, cmap="viridis", norm=LogNorm())
            plt.ylabel("Time")
            plt.xlabel("Intensity")
            plt.colorbar(plot, label="Count")
            pdf.savefig()
            plt.close()
