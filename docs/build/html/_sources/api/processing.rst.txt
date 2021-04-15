Processing
-----------------

Processing module has various submodules. Here summary and detailed
documentation of some of the submodules:

Backends
    .. currentmodule:: dexp.processing.backends

    .. autosummary::
        backend.Backend
        best_backend.BestBackend
        cupy_backend.CupyBackend
        numpy_backend.NumpyBackend

Color
    .. currentmodule:: dexp.processing.color

    .. autosummary::
        blend.blend_color_images
        border.add_border
        cairo_utils.get_array_for_cairo_surface
        colormap.rgb_colormap
        conversions.gray2rgba
        crop_resize_pad.crop_resize_pad_color_image
        insert.insert_color_image
        projection.project_image
        projection_legend.depth_color_scale_legend
        scale_bar.insert_scale_bar
        time_stamp.insert_time_stamp

Deconvolution
    .. currentmodule:: dexp.processing.deconvolution

    .. autosummary::
        lr_deconvolution.lucy_richardson_deconvolution

Deskew
    .. currentmodule:: dexp.processing.deskew

    .. autosummary::
        yang_deskew.yang_deskew

Equalise
    .. currentmodule:: dexp.processing.equalise

    .. autosummary::
        equalise_intensity.equalise_intensity

Filters
    .. currentmodule:: dexp.processing.filters

    .. autosummary::
        butterworth_filter.butterworth_filter
        fft_convolve.fft_convolve
        sobel_filter.sobel_filter

Fusion
    .. currentmodule:: dexp.processing.fusion

    .. autosummary::
        dct_fusion.fuse_dct_nd
        dft_fusion.fuse_dft_nd
        tg_fusion.fuse_tg_nd

Registration
    .. currentmodule:: dexp.processing.registration

    .. autosummary::
        sequence.image_stabilisation
        sequence.image_sequence_stabilisation
        sequence_proj.image_stabilisation_proj
        sequence_proj.image_stabilisation_proj_
        translation_2d.register_translation_2d_skimage
        translation_2d.register_translation_2d_dexp
        translation_nd.register_translation_nd
        translation_nd_proj.register_translation_proj_nd
        warp_multiscale_nd.register_warp_multiscale_nd
        warp_nd.register_warp_nd

Restoration
    .. currentmodule:: dexp.processing.restoration

    .. autosummary::
        aap_correction.axis_aligned_pattern_correction
        clean_dark_regions.clean_dark_regions
        dehazing.dehaze
        lipshitz_correction.lipschitz_continuity_correction

Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dexp.processing.backends.backend

.. autoclass:: Backend
    :members:
    :inherited-members:

.. currentmodule:: dexp.processing.backends.best_backend

.. autofunction:: BestBackend

.. currentmodule:: dexp.processing.backends.cupy_backend

.. autoclass:: CupyBackend
    :members:
    :inherited-members:

.. currentmodule:: dexp.processing.backends.numpy_backend

.. autoclass:: NumpyBackend
    :members:
    :inherited-members:

Color
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dexp.processing.color.blend

.. autofunction:: blend_color_images

.. currentmodule:: dexp.processing.color.border

.. autofunction:: add_border

.. currentmodule:: dexp.processing.color.cairo_utils

.. autofunction:: get_array_for_cairo_surface

.. currentmodule:: dexp.processing.color.colormap

.. autofunction:: rgb_colormap

.. currentmodule:: dexp.processing.color.conversions

.. autofunction:: gray2rgba

.. currentmodule:: dexp.processing.color.crop_resize_pad

.. autofunction:: crop_resize_pad_color_image

.. currentmodule:: dexp.processing.color.insert

.. autofunction:: insert_color_image

.. currentmodule:: dexp.processing.color.projection

.. autofunction:: project_image

.. currentmodule:: dexp.processing.color.projection_legend

.. autofunction:: depth_color_scale_legend

.. currentmodule:: dexp.processing.color.scale_bar

.. autofunction:: insert_scale_bar

.. currentmodule:: dexp.processing.color.time_stamp

.. autofunction:: insert_time_stamp

Deconvolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dexp.processing.deconvolution.lr_deconvolution

.. autofunction:: lucy_richardson_deconvolution

Deskew
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dexp.processing.deskew.yang_deskew

.. autofunction:: yang_deskew

Equalise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dexp.processing.equalise.equalise_intensity

.. autofunction:: equalise_intensity

Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dexp.processing.filters.butterworth_filter

.. autofunction:: butterworth_filter

.. currentmodule:: dexp.processing.filters.fft_convolve

.. autofunction:: fft_convolve

.. currentmodule:: dexp.processing.filters.sobel_filter

.. autofunction:: sobel_filter


Fusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dexp.processing.fusion.dct_fusion

.. autofunction:: fuse_dct_nd

.. currentmodule:: dexp.processing.fusion.dft_fusion

.. autofunction:: fuse_dft_nd

.. currentmodule:: dexp.processing.fusion.tg_fusion

.. autofunction:: fuse_tg_nd

Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dexp.processing.registration.sequence

.. autofunction:: image_stabilisation
.. autofunction:: image_sequence_stabilisation

.. currentmodule:: dexp.processing.registration.sequence_proj

.. autofunction:: image_stabilisation_proj
.. autofunction:: image_stabilisation_proj_

.. currentmodule:: dexp.processing.registration.translation_2d

.. autofunction:: register_translation_2d_skimage
.. autofunction:: register_translation_2d_dexp

.. currentmodule:: dexp.processing.registration.translation_nd

.. autofunction:: register_translation_nd

.. currentmodule:: dexp.processing.registration.translation_nd_proj

.. autofunction:: register_translation_proj_nd

.. currentmodule:: dexp.processing.registration.warp_multiscale_nd

.. autofunction:: register_warp_multiscale_nd

.. currentmodule:: dexp.processing.registration.warp_nd

.. autofunction:: register_warp_nd

Restoration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dexp.processing.restoration.aap_correction

.. autofunction:: axis_aligned_pattern_correction

.. currentmodule:: dexp.processing.restoration.clean_dark_regions

.. autofunction:: clean_dark_regions

.. currentmodule:: dexp.processing.restoration.dehazing

.. autofunction:: dehaze

.. currentmodule:: dexp.processing.restoration.lipshitz_correction

.. autofunction:: lipschitz_continuity_correction
