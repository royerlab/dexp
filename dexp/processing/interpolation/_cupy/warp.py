import cupy

from dexp.processing.backends.backend import Backend


def _warp_1d_cupy(backend: Backend,
                  image,
                  vector_field,
                  internal_dtype=cupy.float16):
    raise NotImplemented("Cupy-based 1D warping not hyet implemented.")


def _warp_2d_cupy(backend: Backend,
                  image,
                  vector_field,
                  internal_dtype=cupy.float16):
    source = r'''
                extern "C"{
                __global__ void copyKernel(float* warped,
                                           cudaTextureObject_t input 
                                           cudaTextureObject_t vector_field,
                                           int width, 
                                           int height)
                {
                    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
                    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

                    if (x < width && y < height)
                    {
                        # coordinates in coord-normalised vector_field texture:
                        float u = x/width;
                        float v = y/height;
                        
                        # Obtain linearly interpolated vector at (u,v):
                        vector = tex2D<float2>(vector_field, u, v);
                        
                        # Obtain the shifted coordinate of the source voxel: 
                        float sx = x + vector.x;
                        float sy = y + vector.y;
                        
                        # Sample source image for voxel value:
                        float value = tex2D<float>(vector_field, sx, sy);
                        
                        # Store interpolated value:
                        warped[y * width + x] = value;
                        
                        #TODO: supersampling would help in regions for which warping misses voxels in the source image,
                        better: adaptive supersampling would automatically use the vector field divergence to determine where
                        to super sample and by how much.  
                    }
                }
                }
                '''

    # set up a texture object

    texobj, arr2 = create_cuda_texture(shape=(width, height),
                                       num_channels=4,
                                       sampling_mode='nearest',
                                       dtype=cupy.float32)

    # allocate input/output arrays
    tex_data = cupy.arange(width * height * 4, dtype=cupy.float32).reshape(height, width * 4)
    real_output = cupy.zeros_like(tex_data)
    expected_output = cupy.zeros_like(tex_data)
    arr2.copy_from(tex_data)
    arr2.copy_to(expected_output)

    # get the kernel, which copies from texture memory
    kernel = cupy.RawKernel(source, 'copyKernel')

    # launch it
    block_x = 4
    block_y = 4
    grid_x = (width + block_x - 1) // block_x
    grid_y = (height + block_y - 1) // block_y
    kernel((grid_x, grid_y), (block_x, block_y), (real_output, texobj, width, height))



def _warp_3d_cupy(backend: Backend,
                  image,
                  vector_field,
                  internal_dtype=cupy.float16):
    raise NotImplemented("Cupy-based 1D warping not hyet implemented.")
