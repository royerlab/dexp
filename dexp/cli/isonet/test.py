from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt


from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, plot_some, axes_dict
from csbdeep.io import save_training_data
from csbdeep.data import RawData, create_patches
from csbdeep.data.transform import anisotropic_distortions


download_and_extract_zip_file (
    url       = 'http://csbdeep.bioimagecomputing.com/example_data/retina.zip',
    targetdir = 'data',
)

x = imread('data/retina/cropped_farred_RFP_GFP_2109175_2color_sub_10.20.tif')
subsample = 10.2
print('image size         =', x.shape)
print('Z subsample factor =', subsample)

plt.figure(figsize=(16,15))
plot_some(np.moveaxis(x,1,-1)[[5,-5]],
          title_list=[['XY slice','XY slice']],
          pmin=2,pmax=99.8);

plt.figure(figsize=(16,15))
plot_some(np.moveaxis(np.moveaxis(x,1,-1)[:,[50,-50]],1,0),
          title_list=[['XZ slice','XZ slice']],
          pmin=2,pmax=99.8, aspect=subsample);



raw_data = RawData.from_folder (
    basepath    = 'data',
    source_dirs = ['retina'],
    target_dir  = 'retina',
    axes        = 'ZCYX',
)

anisotropic_transform = anisotropic_distortions (
    subsample = 10.2,
    psf       = np.ones((3,3))/9, # use the actual PSF here
    psf_axes  = 'YX',
)

X, Y, XY_axes = create_patches (
    raw_data            = raw_data,
    patch_size          = (1,2,128,128),
    n_patches_per_image = 512,
    transforms          = [anisotropic_transform],
)


assert X.shape == Y.shape
print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)


z = axes_dict(XY_axes)['Z']
X = np.take(X,0,axis=z)
Y = np.take(Y,0,axis=z)
XY_axes = XY_axes.replace('Z','')


assert X.shape == Y.shape
print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)

save_training_data('data/my_training_data.npz', X, Y, XY_axes)

for i in range(2):
    plt.figure(figsize=(16,4))
    sl = slice(8*i, 8*(i+1))
    plot_some(np.moveaxis(X[sl],1,-1),np.moveaxis(Y[sl],1,-1),title_list=[np.arange(sl.start,sl.stop)])
    plt.show()



## TRAINING:



import numpy as np
import matplotlib.pyplot as plt


from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, IsotropicCARE


(X,Y), (X_val,Y_val), axes = load_training_data('data/my_training_data.npz', validation_split=0.1, verbose=True)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

plt.figure(figsize=(12,5))
plot_some(X_val[:5],Y_val[:5])
plt.suptitle('5 example validation patches (top row: source, bottom row: target)');



config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=30)
print(config)
vars(config)



model = IsotropicCARE(config, 'my_model', basedir='models')

history = model.train(X,Y, validation_data=(X_val,Y_val))

print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);



















