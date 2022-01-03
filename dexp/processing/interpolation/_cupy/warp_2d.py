import cupy

from dexp.utils.backends import Backend
from dexp.utils.backends._cupy.texture.texture import create_cuda_texture


def _warp_2d_cupy(image, vector_field, mode, block_size: int = 16):
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
                __global__ void warp_2d(float* warped_image,
                                        cudaTextureObject_t input_image,
                                        cudaTextureObject_t vector_field,
                                        int width,
                                        int height)
                {
                    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
                    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

                    if (x < width && y < height)
                    {
                        // coordinates in coord-normalised vector_field texture:
                        float u = float(x)/width;
                        float v = float(y)/height;

                        // Obtain linearly interpolated vector at (u,v):
                        float2 vector = tex2D<float2>(vector_field, u, v);

                        // Obtain the shifted coordinates of the source voxel:
                        // flip axis order to match numpy order
                        float sx = 0.5f + float(x) - vector.y;
                        float sy = 0.5f + float(y) - vector.x;

                        // Sample source image for voxel value:
                        float value = tex2D<float>(input_image, sx, sy);

                        //printf("(%f, %f)=%f\n", sx, sy, value);

                        // Store interpolated value:
                        warped_image[y * width + x] = value;

                        // TODO: supersampling would help in regions for which warping misses voxels in
                        // the source image.
                        // improve: adaptive supersampling would automatically use the vector field divergence to
                        // determine where to super sample and by how much.
                    }
                }
                }
                """

    if image.ndim != 2 or vector_field.ndim != 3:
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
        vector_field, num_channels=2, normalised_coords=True, sampling_mode="linear", address_mode="clamp"
    )

    # Set up resulting image:
    warped_image = xp.empty(shape=image.shape, dtype=image.dtype)

    # get the kernel, which copies from texture memory
    warp_2d_kernel = cupy.RawKernel(source, "warp_2d")

    # launch kernel
    height, width = image.shape

    grid_x = (width + block_size - 1) // block_size
    grid_y = (height + block_size - 1) // block_size
    warp_2d_kernel(
        (grid_x, grid_y), (block_size,) * 2, (warped_image, input_image_tex, vector_field_tex, width, height)
    )

    del input_image_tex, input_image_cudarr, vector_field_tex, vector_field_cudarr

    return warped_image
