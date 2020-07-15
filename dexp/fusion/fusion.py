import gc

import cupy
# Disable memory pool for device memory (GPU)
from napari import Viewer, gui_qt
from numpy import newaxis, real

cupy.cuda.set_allocator(None)
# Disable memory pool for pinned memory (CPU).
cupy.cuda.set_pinned_memory_allocator(None)

import math
import time
import numexpr
import numpy
import torch
from scipy.ndimage import center_of_mass
from dexp.fusion.base_fusion import BaseFusion


class SimpleFusion(BaseFusion):

    def __init__(self, split_point_x=0.5, split_point_z=0.5, smoothness_x=60, smoothness_z=12, backend='cupy', dtype=numpy.float16):

        super().__init__()


        self.backend = backend
        self.device = self.backend.split('-')[1] if '-' in self.backend else None
        self.dtype = dtype
        self.split_point_x = split_point_x
        self.split_point_z = split_point_z
        self.smoothness_x = smoothness_x
        self.smoothness_z = smoothness_z

        self.blending_shape = None

        self.butterworth_filter = None

        #_convert_to_backend
    def _cb(self, array):
        if 'pytorch' in self.backend:
            if torch.is_tensor(array):
                return array
            else:
                return torch.tensor(array.astype(self.dtype, copy=False), requires_grad=False, device=self.device)
        elif 'cupy' in self.backend:
            if cupy.get_array_module(array) == cupy:
                return array
            else:
                return cupy.asarray(array.astype(self.dtype, copy=False))
        elif 'numpy' in self.backend:
            return array.astype(self.dtype, copy=False)

    #_convert_to_numpy (from backend)
    def _cn(self, array):
        if 'pytorch' in self.backend:
            if torch.is_tensor(array):
                return array.cpu().detach().numpy().astype(self.dtype, copy=False)
            else:
                return array.astype(self.dtype, copy=False)
        elif 'cupy' in self.backend:
            if cupy.get_array_module(array) == cupy:
                return cupy.asnumpy(array.astype(self.dtype, copy=False))
            else:
                return array.astype(self.dtype, copy=False)
        elif 'numpy' in self.backend:
            return array.astype(self.dtype, copy=False)

    def _get_blending_map_z(self, shape, split_point_z, smoothness):
        length = shape[0]
        spz = int(split_point_z * length)
        profile = numpy.fromfunction(lambda z: 1.0 - 0.5 * (1 + (((z - spz) / smoothness) / (1.0 + ((z - spz) / smoothness) ** 2) ** 0.5)), shape=(length,), dtype=self.dtype)
        profile = profile.astype(self.dtype, copy=False)
        blending_map = profile[...,newaxis, newaxis]
        return blending_map

    def _get_blending_map_x(self, shape, split_point_x, smoothness):
        length = shape[-1]
        spx = int(split_point_x * length)
        profile = numpy.fromfunction(lambda x: 1.0 - 0.5 * (1 + (((x - spx) / smoothness) / (1.0 + ((x - spx) / smoothness) ** 2) ** 0.5)), shape=(length,), dtype=self.dtype)
        profile = profile.astype(self.dtype, copy=False)
        blending_map = profile[newaxis, newaxis, ...]
        return blending_map

    def _initialise_blending_maps(self, shape):

        if self.blending_shape != shape:
            self.blending_shape = shape
            print(f"Creating blend weights...")
            blending_x = self._get_blending_map_x(shape, self.split_point_x, self.smoothness_x)
            blending_z = self._get_blending_map_z(shape, self.split_point_z, self.smoothness_z)
            self.blending_CxL0 = blending_x.astype(self.dtype, copy=False)
            self.blending_CxL1 = (1.0 - blending_x).astype(self.dtype, copy=False)
            self.blending_C0Lx = blending_z.astype(self.dtype, copy=False)
            self.blending_C1Lx = (1.0 - blending_z).astype(self.dtype, copy=False)
            print(f"Done creating blend weights...")


    def equalise_intensity(self, image1, image2, zero_level=90, percentile=0.999, reduction=32, as_numpy=False):

        image1 = self._cb(image1)
        image2 = self._cb(image2)

        xp = cupy.get_array_module(image1)

        strided_image1 = image1.ravel()[::reduction]
        strided_image2 = image2.ravel()[::reduction]

        highvalue1 = xp.percentile(strided_image1.astype(numpy.float32), q=percentile*100)
        highvalue2 = xp.percentile(strided_image2.astype(numpy.float32), q=percentile*100)

        mask1 = strided_image1 >= highvalue1
        mask2 = strided_image2 >= highvalue2

        mask = mask1 * mask2

        highvalues1 = strided_image1[mask]
        highvalues2 = strided_image2[mask]

        ratios = highvalues1/highvalues2

        median_ratio = xp.percentile(ratios.astype(numpy.float32), q=50)

        if zero_level!=0:
            image1 -= zero_level
            image2 -= zero_level

        image1.clip(0, math.inf, out=image1)
        image2.clip(0, math.inf, out=image2)

        if median_ratio>1:
            image2 *= median_ratio
        else:
            image1 *= (1/median_ratio)

        image1 = image1.astype(self.dtype, copy=False)
        image2 = image2.astype(self.dtype, copy=False)

        if as_numpy:
            image1 = self._cn(image1)
            image2 = self._cn(image2)

        return image1, image2

    def fuse_lightsheets(self, CxL0, CxL1, as_numpy=False):

        self._initialise_blending_maps(CxL0.shape)

        bCxL0 = self.blending_CxL0
        bCxL1 = self.blending_CxL1

        if 'pytorch' in self.backend:
            with torch.no_grad():
                fused = self._cb(bCxL0) * self._cb(CxL0) + self._cb(bCxL1) * self._cb(CxL1)
        elif 'cupy' in self.backend:
            CxL0 = self._cb(CxL0)
            CxL1 = self._cb(CxL1)
            CxL0 *= self._cb(bCxL0)
            CxL1 *= self._cb(bCxL1)
            fused = CxL0+CxL1

        elif 'numpy' in self.backend:
            fused = numexpr.evaluate("bCxL0 * CxL0 + "
                                     "bCxL1 * CxL1 ")
        if as_numpy:
            fused = self._cn(fused)

        return fused


    def _find_shift(self, a, b, max_range=128, fine_window_radius=4, decimate=16, percentile=99.9):

        xp = cupy.get_array_module(a)

        a = self._cb(a)
        b = self._cb(b)

        # We compute the phase correlation:
        raw_correlation = self._phase_correlation(a, b)

        correlation = raw_correlation

        # We estimate the noise floor of the correlation:
        max_ranges = tuple(max(0, min(max_range, s-2*max_range)) for s in correlation.shape)
        print(f"max_ranges={max_ranges}")
        empty_region = correlation[tuple(slice(r, s-r) for r,s in zip(max_ranges, correlation.shape))].copy()
        noise_floor_level = xp.percentile(empty_region.ravel()[::decimate].astype(numpy.float32), q=percentile)
        print(f"noise_floor_level={noise_floor_level}")

        # we use that floor to clip anything below:
        correlation = correlation.clip(noise_floor_level, math.inf) - noise_floor_level

        # We roll the array and crop it to restrict ourself to the search region:
        correlation = xp.roll(correlation, shift=max_range, axis=tuple(range(a.ndim)))
        correlation = correlation[(slice(0, 2 * max_range),) * a.ndim]

        # denoise cropped corelation image:
        #correlation = gaussian_filter(correlation, sigma=sigma, mode='wrap')

        # We use the max as quickly computed proxy for the real center:
        rough_shift = xp.unravel_index(
            xp.argmax(correlation, axis=None), correlation.shape
        )

        print(f"rough_shift= {rough_shift}")

        # We crop further to facilitate center-of-mass estimation:
        cropped_correlation = correlation[
            tuple(
                slice(max(0, int(s) - fine_window_radius), min(d, int(s) + fine_window_radius))
                for s, d in zip(rough_shift, correlation.shape)
            )
        ]
        print(f"cropped_correlation.shape = {cropped_correlation.shape}")

        # with gui_qt():
        #     viewer = Viewer()
        #     viewer.add_image(self._cn(a), name='a')
        #     viewer.add_image(self._cn(b), name='b')
        #     viewer.add_image(self._cn(raw_correlation), name='raw_correlation')
        #     viewer.add_image(self._cn(correlation), name='correlation')
        #     viewer.add_image(self._cn(cropped_correlation), name='cropped_correlation')

        # We compute the signed rough shift
        signed_rough_shift = xp.array(rough_shift) - max_range
        print(f"signed_rough_shift= {signed_rough_shift}")

        signed_rough_shift = self._cn(signed_rough_shift)
        cropped_correlation = self._cn(cropped_correlation)

        # We compute the center of mass:
        # We take the square to squash small values far from the maximum that are likely noisy...
        signed_com_shift = (
                numpy.array(center_of_mass(cropped_correlation ** 2))
                - fine_window_radius
        )
        print(f"signed_com_shift= {signed_com_shift}")

        # The final shift is the sum of the rough sight plus the fine center of mass shift:
        shift = list(signed_rough_shift + signed_com_shift)

        print(f"shift = {shift}")

        return shift, correlation



    def _phase_correlation(self, image, reference_image, as_numpy=False):

        xp = cupy.get_array_module(image)
        G_a = xp.fft.fftn(image).astype(numpy.complex64, copy=False)
        G_b = xp.fft.fftn(reference_image).astype(numpy.complex64, copy=False)
        conj_b = xp.conj(G_b)
        R = G_a * conj_b
        R /= xp.absolute(R)
        r = xp.fft.ifftn(R).real.astype(self.dtype, copy=False)

        if as_numpy:
            r = self._cn(r)

        return r


    def register_stacks(self, C0Lx, C1Lx, overlapp=96, crop_x=128, crop_y=512, shifts=None, as_numpy=False):

        C0Lx = self._cb(C0Lx)
        C1Lx = self._cb(C1Lx)

        # We compute the registration parameters on the central overlap region:
        length = C0Lx.shape[0]
        begin_z = length//2-overlapp//2
        end_z   = length//2+overlapp//2

        C0Lx_crop = C0Lx[begin_z:end_z, crop_y:-crop_y, crop_x:-crop_x]
        C1Lx_crop = C1Lx[begin_z:end_z, crop_y:-crop_y, crop_x:-crop_x]

        if shifts is None:
            shifts, _ = self._find_shift(C0Lx_crop, C1Lx_crop, max_range=overlapp//2)

        for i, shift in enumerate(shifts):
            integral_shift = int(round(shift))
            xp = cupy.get_array_module(C1Lx)
            C1Lx = xp.roll(C1Lx, shift=integral_shift, axis=i)

        if as_numpy:
            C0Lx, C1Lx = self._cn(C0Lx), self._cn(C1Lx)

        return C0Lx, C1Lx, shifts

    def fuse_cameras(self, C0Lx, C1Lx, as_numpy=False):

        self._initialise_blending_maps(C0Lx.shape)

        bC0Lx = self.blending_C0Lx
        bC1Lx = self.blending_C1Lx

        if 'pytorch' in self.backend:
            with torch.no_grad():
                fused = self._cb(bC0Lx) * self._cb(C0Lx) + self._cb(bC1Lx) * self._cb(C1Lx)
        elif 'cupy' in self.backend:
            C0Lx = self._cb(C0Lx)
            C1Lx = self._cb(C1Lx)
            C0Lx *= self._cb(bC0Lx)
            C1Lx *= self._cb(bC1Lx)
            fused = C0Lx + C1Lx
        elif 'numexpr' in self.backend:
            fused = numexpr.evaluate("bC0Lx * C0Lx + "
                                     "bC1Lx * C1Lx ")
        if as_numpy:
            fused = self._cn(fused)

        return fused


    def butterworth(self, filter_shape, cutoffs, n=3):

        lz, ly, lx = filter_shape
        cz, cy, cx = cutoffs

        x =  numpy.linspace(-0.5, 0.5, lx)
        y =  numpy.linspace(-0.5, 0.5, ly)
        z =  numpy.linspace(-0.5, 0.5, lz)

        # An array with every pixel = radius relative to center
        radius = numpy.sqrt(((x/cx)**2)[newaxis, newaxis, :] + ((y / cy) ** 2)[newaxis, :, newaxis] + ((z / cz) ** 2)[:, newaxis, newaxis])

        filter = 1 / (1.0 + radius**(2*n))

        from numpy.fft import fftshift
        from numpy.fft import ifftn
        from numpy.fft import ifftshift
        kernel = fftshift(real(ifftn(ifftshift(filter))))

        kernel = kernel/kernel.sum()

        return kernel.astype(numpy.float32)


    def butterworth_noise_filter(self, image, filter_shape, cutoffs, n=3, as_numpy=False):

        if self.butterworth_filter is None:
            butterworth_filter_numpy = self.butterworth(filter_shape, cutoffs, n)
            self.butterworth_filter = self._cb(butterworth_filter_numpy)

        bimage = self._cb(image)

        from napari import gui_qt, Viewer
        with gui_qt():
            viewer = Viewer()
            viewer.add_image(butterworth_filter_numpy, name='filter')

        from cupyx.scipy.ndimage import convolve
        filtered_image = convolve(bimage, self.butterworth_filter, mode='reflect')

        from napari import gui_qt, Viewer
        with gui_qt():
            viewer = Viewer()
            viewer.add_image(self._cn(filtered_image), name='filter')

        if as_numpy:
            filtered_image = self._cn(filtered_image)

        return filtered_image


    def fuse_2I2D(self, C0L0, C0L1, C1L0, C1L1, shifts=None, zero_level=100, filter=False, as_numpy=False):

        start = time.time()
        print(f"equalise_intensity...")
        C0L0, C0L1 = self.equalise_intensity(C0L0, C0L1, zero_level=zero_level, as_numpy=False)
        gc.collect()
        print(f"fuse_lightsheets...")
        C0lx = self.fuse_lightsheets(C0L0, C0L1, as_numpy=False)
        del C0L0
        del C0L1
        gc.collect()

        print(f"equalise_intensity...")
        C1L0, C1L1 = self.equalise_intensity(C1L0, C1L1, zero_level=zero_level,  as_numpy=False)
        gc.collect()
        print(f"fuse_lightsheets...")
        C1lx = self.fuse_lightsheets(C1L0, C1L1, as_numpy=False)
        del C1L0
        del C1L1
        gc.collect()

        print(f"equalise_intensity...")
        C0lx, C1lx = self.equalise_intensity(C0lx, C1lx, zero_level=0, as_numpy=False)
        gc.collect()
        print(f"register_stacks...")
        C0lx, C1lx, shifts = self.register_stacks(C0lx, C1lx, shifts=shifts, as_numpy=False)
        gc.collect()
        print(f"fuse_cameras...")
        CxLx = self.fuse_cameras(C0lx, C1lx, as_numpy=False)
        del C0lx
        del C1lx
        gc.collect()
        stop = time.time()
        print(f"Elapsed fusion time:  {stop-start} for backend: {self.backend} ")

        if filter:
            print(f"Filter output using a Butterworth filter")
            CxLx = self.butterworth_noise_filter(CxLx, filter_shape=(17, 17, 17), cutoffs=(0.9, 0.9, 0.9))

        if as_numpy:
            CxLx = self._cn(CxLx)

        return CxLx, shifts



