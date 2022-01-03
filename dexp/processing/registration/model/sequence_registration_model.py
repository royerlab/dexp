import json
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from dexp.processing.registration.model.pairwise_registration_model import (
    PairwiseRegistrationModel,
)
from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.utils import xpArray
from dexp.utils.backends import Backend


class SequenceRegistrationModel:
    def __init__(self, model_list: List[PairwiseRegistrationModel] = [], force_numpy: bool = True):

        """Instantiates a sequence-registration-model.
        A sequence registration model describes how to register jointly each n-1 D slice of a nD image.
        In effect, this object contains a sequence of registration models that describe how to
        transform each image in the sequence to the registered image that maximises registration
        coherence across the whole sequence.

        Parameters
        ----------
        model_list : list of registration models that
        force_numpy : when creating this object, you have the option of forcing the use of
            numpy array instead of the current backend arrays.
        integral : True if shifts are snapped to integer values, False otherwise

        """
        super().__init__()

        self.model_list: List[PairwiseRegistrationModel] = model_list
        if force_numpy:
            self.to_numpy()

    def append(self, model: PairwiseRegistrationModel):
        self.model_list.append(model)

    def __iadd__(self, models: Sequence[PairwiseRegistrationModel]):
        self.model_list += models

    def __len__(self):
        return self.model_list.__len__()

    def __getitem__(self, item) -> PairwiseRegistrationModel:
        return self.model_list.__getitem__(item)

    def __str__(self):
        return f"SequenceRegistrationModel(length={len(self.model_list)})"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.model_list == other.model_list
        else:
            return False

    def to_json(self) -> str:
        models_json = list(model.to_json() for model in self.model_list)
        return json.dumps({"type": "translation_sequence", "model_list": models_json})

    def to_numpy(self) -> "SequenceRegistrationModel":
        self.model_list = list(model.to_numpy() for model in self.model_list)
        return self

    def overall_confidence(self) -> float:
        confidences = list(model.overall_confidence() for model in self.model_list)
        return float(np.median(confidences))

    def padding(self):
        padding_list = list(model.padding() for model in self.model_list)
        overall_padding = padding_list[0]
        for padding in padding_list:
            overall_padding = tuple((max(op[0], p[0]), max(op[1], p[1])) for op, p in zip(overall_padding, padding))
        return tuple(overall_padding)

    def padded_shape(self, shape: Tuple[int, ...]):
        new_shape = tuple(s + pl + pr for s, (pl, pr) in zip(shape, self.padding()))
        return new_shape

    def apply_sequence(
        self, image, axis: int, pad_width: Optional[Union[float, Tuple[Tuple[float, float], ...]]] = None, **kwargs
    ) -> xpArray:
        """Applies this sequence registration model to the an image sequence of given sequence axis.
        A new registered image is returned.

        Parameters
        ----------
        image: image sequence to register must be n+1 dimensional (1 dimension is the sequence axis)
        axis: sequence axis
        pad_width: Can be: (i) list of pad widths for all image edges : ((pad_left, pad_right),(...,...),..)
            (see numpy.pad pad_widths parameter), (ii) '0' (zero) for no padding, or (iii) None for automatic padding.
        **kwargs: parameters passthrough to the apply method of the pairwise registration model

        Returns
        -------
        stabilised image sequence

        """

        xp = Backend.get_xp_module()

        # move axis to first position, makes everything easier:
        if axis != 0:
            image = xp.moveaxis(image, axis, 0)

        if pad_width is None:
            # automatic padding
            image = xp.pad(image, pad_width=((0, 0),) + self.padding())

        elif pad_width != 0:
            # custom padding
            image = xp.pad(image, pad_width=((0, 0),) + tuple(pad_width))
        else:
            # no padding
            pass

        registered_image = xp.stack(
            self.apply(image[index], index=index, **kwargs) for index in range(image.shape[axis])
        )

        # put back axis where it belongs, if necessary:
        if axis != 0:
            registered_image = xp.moveaxis(registered_image, 0, axis)

        return registered_image

    def apply(self, image, index: int, pad: bool = False, **kwargs) -> xpArray:
        """Applies this sequence registration model to the image at a given index.
        A new registered image is returned.


        Parameters
        ----------
        image: image to register
        index: index of image in sequence
        pad : pads the image before registration
        **kwargs : parameters passthrough to the apply method of the pairwise registration model

        Returns
        -------
        image_reg: registered image against all others in the sequence

        """
        xp = Backend.get_xp_module()
        model = self.model_list[index]

        if pad:
            padding = self.padding()
            image = xp.pad(image, pad_width=padding)
            return model.apply(image, **kwargs)

        else:
            return model.apply(image, **kwargs)

    def plot(self, path: str):
        import matplotlib.pyplot as plt

        length = len(self.model_list)
        ndim = self.model_list[0].shift_vector.shape[0]
        x = np.arange(0, length)
        y = np.zeros((length, ndim))
        for i, m in enumerate(self.model_list):
            y[i] = m.shift_vector

        fig, ax = plt.subplots()  # Create a figure and an axes.
        for axis in range(ndim):
            ax.plot(x, y[:, axis], label=f"axis {axis}")  # Plot some data on the axes.

        ax.set_xlabel("time points")  # Add an x-label to the axes.
        ax.set_ylabel("shift")  # Add a y-label to the axes.
        ax.set_title(f"{path} shifts")  # Add a title to the axes.
        ax.legend()  # Add a legend.

        plt.savefig(path + "_shifts.pdf")

    def reduce(self, step: int) -> "SequenceRegistrationModel":
        if not isinstance(self.model_list[0], TranslationRegistrationModel):
            raise NotImplementedError

        new_model_list = []
        for i in range(0, len(self.model_list), step):
            new_model = self.model_list[i].copy()
            for j in range(i + 1, min(len(self.model_list), i + step)):
                centered_model = self.model_list[j] - self.model_list[j - 1]
                new_model += centered_model
            new_model_list.append(new_model)

        return SequenceRegistrationModel(new_model_list)

    def total_displacement(self) -> List[Tuple[int]]:
        if not isinstance(self.model_list[0], TranslationRegistrationModel):
            raise NotImplementedError

        lower_disp = np.zeros_like(self.model_list[0].shift_vector)
        upper_disp = np.zeros_like(self.model_list[0].shift_vector)
        for model in self:
            lower_disp = np.minimum(lower_disp, model.shift_vector)
            upper_disp = np.maximum(upper_disp, model.shift_vector)

        return [(l.item(), u.item()) for l, u in zip(lower_disp, upper_disp)]
