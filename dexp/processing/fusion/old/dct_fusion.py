import numpy
import torch
from scipy.fft import dctn, idctn
from torch_dct import dct_3d, idct_3d

from dexp.processing.fusion import SimpleFusion


class DCTFusion(SimpleFusion):

    def __init__(self, device='cpu', width=0.2, cutofflow=0.3, blending=False, **kwargs):

        super().__init__(**kwargs)

        self.width = width
        self.blending = blending
        self.device = device
        self.cutofflow = cutofflow

    def _fuse_dct(self, array1, array2, cutoff):

        c = cutoff
        d, h, w = array1.shape
        cz, cy, cx = int(d * c), int(h * c), int(w * c)

        if 'pytorch' in self.backend:
            with torch.no_grad():
                array1_dct = dct_3d(array1)
                array2_dct = dct_3d(array2)

                max_dct = torch.max(array1_dct, array2_dct)
                fused_dct = max_dct

                min_dct = torch.min(array1_dct, array2_dct)

                fused_dct[0:cz, 0:cy, 0:cx] = min_dct[0:cz, 0:cy, 0:cx]

                fused = idct_3d(fused_dct)

        elif 'numexpr' in self.backend:
            array1_dct = dctn(array1)
            array2_dct = dctn(array2)

            max_dct = numpy.max(array1_dct, array2_dct)
            fused_dct = max_dct

            min_dct = numpy.min(array1_dct, array2_dct)

            fused_dct[0:cz, 0:cy, 0:cx] = min_dct[0:cz, 0:cy, 0:cx]

            fused = idctn(fused_dct)

        return fused

    def fuse_lightsheets(self, CxL0, CxL1, as_numpy=True):

        fused = super().fuse_illumination_views(CxL0, CxL1, as_numpy=False)
        fused = self._cb(fused)

        CxL0 = self._cb(CxL0)
        CxL1 = self._cb(CxL1)

        L0_blend_map = self._get_blending_map_x(CxL0.shape, self.split_point_x + self.width / 2, self.smoothness_x)
        L1_blend_map = 1 - self._get_blending_map_x(CxL0.shape, 1 - self.split_point_x - self.width / 2, self.smoothness_x)

        blend_map = L0_blend_map * L1_blend_map
        blend_map = self._cb(blend_map)

        width = CxL0.shape[-1]
        x_begin = int((self.split_point_x - self.width / 2) * width) - self.smoothness_x
        x_end = int((self.split_point_x + self.width / 2) * width) + self.smoothness_x
        x_begin = max(0, x_begin)
        x_end = min(0, width - 1)

        CxL0_cs = CxL0[:, :, x_begin:x_end]
        CxL1_cs = CxL1[:, :, x_begin:x_end]
        blend_map_cs = blend_map[:, :, x_begin:x_end]

        dct_fused_cs = self._fuse_dct(CxL0_cs, CxL1_cs, self.cutofflow)

        fused[:, :, x_begin:x_end] = fused[:, :, x_begin:x_end] * (1 - blend_map_cs) + dct_fused_cs * (blend_map_cs)

        if as_numpy:
            fused = self._cn(fused)

        return fused

    def fuse_cameras(self, C0Lx, C1Lx, as_numpy=True):

        fused = super().fuse_detection_views(C0Lx, C1Lx, as_numpy=False)
        fused = self._cb(fused)

        C0Lx = self._cb(C0Lx)
        C1Lx = self._cb(C1Lx)

        C0_blend_map = self._get_blending_map_z(C0Lx.shape, self.split_point_z + self.width / 2, self.smoothness_x)
        C1_blend_map = 1 - self._get_blending_map_z(C1Lx.shape, 1 - self.split_point_z - self.width / 2, self.smoothness_x)

        blend_map = C0_blend_map * C1_blend_map
        blend_map = self._cb(blend_map)

        width = C0Lx.shape[-1]
        z_begin = int((self.split_point_z - self.width / 2) * width) - self.smoothness_x
        z_end = int((self.split_point_z + self.width / 2) * width) + self.smoothness_x
        z_begin = max(0, z_begin)
        z_end = min(0, width - 1)

        C0Lx_cs = C0Lx[z_begin:z_end, :, :]
        C1Lx_cs = C1Lx[z_begin:z_end, :, :]
        blend_map_cs = blend_map[z_begin:z_end, :, :]

        dct_fused_cs = self._fuse_dct(C0Lx_cs, C1Lx_cs, self.cutofflow)

        fused[z_begin:z_end, :, :] = fused[z_begin:z_end, :, :] * (1 - blend_map_cs) + dct_fused_cs * (blend_map_cs)

        if as_numpy:
            fused = self._cn(fused)

        return fused
