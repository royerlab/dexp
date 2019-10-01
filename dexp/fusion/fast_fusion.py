import numexpr

import numpy

from dexp.fusion.base_fusion import BaseFusion


class FastFusion(BaseFusion):

    def __init__(self, shape, split_point_x=0.5, split_point_z=0.5, smoothness=60):


        spx = int(split_point_x*shape[-1])
        spz = int(split_point_z*shape[0])
        smoothx = smoothness
        smoothz = smoothness

        print(f"Creating blend weights...")
        blending_x = numpy.fromfunction(lambda z, y, x: 1.0 - 0.5*(1+(((x-spx)/smoothx) / (1.0 + ((x-spx)/smoothx) ** 2) ** 0.5)), shape=shape, dtype=numpy.float32)
        blending_z = numpy.fromfunction(lambda z, y, x: 1.0 - 0.5*(1+(((z-spz)/smoothz) / (1.0 + ((z-spz)/smoothz) ** 2) ** 0.5)), shape=shape, dtype=numpy.float32)

        self.blending_C0L0 = blending_z*blending_x
        self.blending_C0L1 = blending_z*(1-blending_x)
        self.blending_C1L0 = (1-blending_z)*(blending_x)
        self.blending_C1L1 = (1-blending_z)*(1-blending_x)


    def fuse(self, C0L0, C0L1, C1L0, C1L1):

        bC0L0 = self.blending_C0L0
        bC0L1 = self.blending_C0L1
        bC1L0 = self.blending_C1L0
        bC1L1 = self.blending_C1L1

        array = numexpr.evaluate("bC0L0 * C0L0 + "
                                 "bC0L1 * C0L1 + "
                                 "bC1L0 * C1L0 + "
                                 "bC1L1 * C1L1")

        return array



