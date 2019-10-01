import numpy as np
import pywt


# This function does the coefficient fusing according to the fusion method
def fuseCoeff(cooef1, cooef2, method):

    if (method == 'mean'):
        cooef = (cooef1 + cooef2) / 2
    elif (method == 'geomean'):
        product = cooef1 * cooef2
        cooef = np.sign(product)*(np.abs(product)) ** 0.5
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    else:
        cooef = []

    return cooef


def wavelet_fusion(image1, image2, mode='mean'):
    ## Fusion algo
    ## mode 'mean', 'min', 'max'

    # First: Do wavelet transform on each image
    wavelet = 'db1'
    cooef1 = pywt.wavedec2(image1[:,:], wavelet)
    cooef2 = pywt.wavedec2(image2[:,:], wavelet)

    # Second: for each level in both image do the fusion according to the desire option
    fusedCooef = []
    for i in range(len(cooef1)):

        # The first values in each decomposition is the apprximation values of the top level
        if(i == 0):
            fusedCooef.append(fuseCoeff(cooef1[0],cooef2[0],mode))
        else:

            # For the rest of the levels we have tuples with 3 coefficients
            c1 = fuseCoeff(cooef1[i][0], cooef2[i][0], mode)
            c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], mode)
            c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], mode)

            fusedCooef.append((c1,c2,c3))

    # Third: After we fused the coefficients we need to transform back to get the image
    fusedImage = pywt.waverec2(fusedCooef, wavelet)

    return fusedImage