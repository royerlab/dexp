import cupy

from dexp.utils.backends import Backend
from dexp.utils.backends._cupy.texture.texture import create_cuda_texture


def _warp_1d_cupy(image, vector_field, mode, block_size: int = 128):
    """

    Parameters
    ----------
    image
    vector_field
    mode
    block_size

    Returns
    -------

    """
    xp = Backend.get_xp_module()
    source = r"""
                extern "C"{
                __global__ void warp_1d(float* warped_image,
                                        cudaTextureObject_t input_image,
                                        cudaTextureObject_t vector_field,
                                        int width)
                {
                    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

                    if (x < width)
                    {
                        // coordinates in coord-normalised vector_field texture:
                        float u = float(x)/width;

                        // Obtain linearly interpolated vector at (u,):
                        float vector = tex1D<float>(vector_field, u);

                        // Obtain the shifted coordinates of the source voxel:
                        float sx = 0.5f + float(x) - vector;

                        // Sample source image for voxel value:
                        float value = tex1D<float>(input_image, sx);

                        //printf("(%f, %f)=%f\n", sx, sy, value);

                        // Store interpolated value:
                        warped_image[x] = value;

                    }
                }
                }
                """

    if image.ndim != 1 or vector_field.ndim != 1:
        raise ValueError("image or vector field has wrong number of dimensions!")

    # set up textures:
    input_image_tex, input_image_cudarr = create_cuda_texture(
        image,
        num_channels=1,
        normalised_coords=False,
        sampling_mode="linear",
        address_mode=mode,
    )

    vector_field_tex, vector_field_cudarr = create_cuda_texture(
        vector_field, num_channels=1, normalised_coords=True, sampling_mode="linear", address_mode="clamp"
    )

    # Set up resulting image:
    warped_image = xp.empty(shape=image.shape, dtype=image.dtype)

    # get the kernel, which copies from texture memory
    warp_1d_kernel = cupy.RawKernel(source, "warp_1d")

    # launch kernel
    (width,) = image.shape

    grid_x = (width + block_size - 1) // block_size
    warp_1d_kernel((grid_x,), (block_size,), (warped_image, input_image_tex, vector_field_tex, width))

    del input_image_tex, input_image_cudarr, vector_field_tex, vector_field_cudarr

    return warped_image
