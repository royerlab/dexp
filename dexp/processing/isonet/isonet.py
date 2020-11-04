import tempfile
import warnings
from os.path import join

import numpy as np
from csbdeep.data import RawData, create_patches, no_background_patches
from csbdeep.data.transform import anisotropic_distortions
from csbdeep.io import load_training_data
from csbdeep.io import save_training_data
from csbdeep.models import Config, IsotropicCARE


class IsoNet:

    def __init__(self, context_name='default', subsampling=5):
        """

        """

        self.context_folder = join(tempfile.gettempdir(), context_name)
        self.training_data_path = join(self.context_folder, "training_data.npz")
        self.model_path = join(self.context_folder, "model")

        self.model = None

        self.subsampling = subsampling

    def prepare(self, image, psf=np.ones((3, 3)) / 9, threshold=0.9):
        print('image size         =', image.shape)
        print('Z subsample factor =', self.subsampling)

        raw_data = RawData.from_arrays(image, image, axes='ZYX')

        anisotropic_transform = anisotropic_distortions(
            subsample=self.subsampling,
            psf=psf,  # use the actual PSF here
            psf_axes='YX',
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, Y, XY_axes = create_patches(
                raw_data=raw_data,
                patch_size=(128, 128),
                n_patches_per_image=512,
                transforms=[anisotropic_transform],
                patch_filter=no_background_patches(threshold=threshold, percentile=100 - 0.1),
            )

        assert X.shape == Y.shape
        print("shape of X,Y =", X.shape)
        print("axes  of X,Y =", XY_axes)

        print("Saving training data...")
        save_training_data(self.training_data_path, X, Y, XY_axes)
        print("Done saving training data.")

    def train(self, max_epochs=100):
        (X, Y), (X_val, Y_val), axes = load_training_data(self.training_data_path, validation_split=0.1, verbose=True)

        n_channel_in, n_channel_out = 1, 1

        config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=30)
        print(config)
        vars(config)

        model = IsotropicCARE(config, 'my_model', basedir=self.model_path)

        history = model.train(X, Y, validation_data=(X_val, Y_val), epochs=max_epochs)

        print(sorted(list(history.history.keys())))

    def apply(self, image, batch_size=8):
        if self.model is None:
            print('Loading model.')
            self.model = IsotropicCARE(config=None, name='my_model', basedir=self.model_path)
        axes = 'ZYX'

        print(f'Applying model to image of shape: {image.shape}...')
        restored = self.model.predict(image, axes, self.subsampling, batch_size=batch_size)
        print('done!')

        return restored
