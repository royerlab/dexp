import cupy

from dexp.processing.backends._cupy.texture.texture import create_cuda_texture
from dexp.processing.backends.backend import Backend


def _warp_1d_cupy(backend: Backend,
                  image,
                  vector_field,
                  internal_dtype=cupy.float16,
                  block_size: int = 128):

    xp = backend.get_xp_module()
    source = r'''
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
                        float sx = float(x) + vector;

                        // Sample source image for voxel value:
                        float value = tex1D<float>(input_image, sx);

                        //printf("(%f, %f)=%f\n", sx, sy, value);

                        // Store interpolated value:
                        warped_image[x] = value;
 
                    }
                }
                }
                '''

    if image.ndim != 1 or vector_field.ndim != 1:
        raise ValueError("image or vector field has wrong number of dimensions!")

    # set up textures:
    input_image_tex = create_cuda_texture(image,
                                          num_channels=1,
                                          normalised_coords=False,
                                          sampling_mode='linear')

    vector_field_tex = create_cuda_texture(vector_field,
                                           num_channels=1,
                                           normalised_coords=True,
                                           sampling_mode='linear')

    # Set up resulting image:
    warped_image = xp.empty_like(image)

    # get the kernel, which copies from texture memory
    warp_1d_kernel = cupy.RawKernel(source, 'warp_1d')

    # launch kernel
    width, = image.shape

    grid_x = (width + block_size - 1) // block_size
    warp_1d_kernel((grid_x,),
                   (block_size,),
                   (warped_image, input_image_tex, vector_field_tex, width))

    return warped_image


def _warp_2d_cupy(backend: Backend,
                  image,
                  vector_field,
                  internal_dtype=cupy.float16,
                  block_size: int = 16):
    xp = backend.get_xp_module()
    source = r'''
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
                        
                        //printf("(%f, %f)=%f\n", sx, sy, value);
                        
                        // Obtain the shifted coordinates of the source voxel: 
                        float sx = float(x) + vector.x;
                        float sy = float(y) + vector.y;
                        
                        // Sample source image for voxel value:
                        float value = tex2D<float>(input_image, sx, sy);
                        
                        
                        
                        // Store interpolated value:
                        warped_image[y * width + x] = value;
                        
                        //TODO: supersampling would help in regions for which warping misses voxels in the source image,
                        //better: adaptive supersampling would automatically use the vector field divergence to determine where
                        //to super sample and by how much.  
                    }
                }
                }
                '''

    if image.ndim != 2 or vector_field.ndim != 3:
        raise ValueError("image or vector field has wrong number of dimensions!")

    # set up textures:
    input_image_tex = create_cuda_texture(image,
                                          num_channels=1,
                                          normalised_coords=False,
                                          sampling_mode='linear')

    vector_field_tex = create_cuda_texture(vector_field,
                                          num_channels=2,
                                          normalised_coords=True,
                                          sampling_mode='linear')

    # Set up resulting image:
    warped_image = xp.empty_like(image)

    # get the kernel, which copies from texture memory
    warp_2d_kernel = cupy.RawKernel(source, 'warp_2d')

    # launch kernel
    width, height = image.shape

    grid_x = (width + block_size - 1) // block_size
    grid_y = (height + block_size - 1) // block_size
    warp_2d_kernel((grid_x, grid_y),
                   (block_size,)*2,
                   (warped_image, input_image_tex, vector_field_tex, width, height))

    return warped_image


def _warp_3d_cupy(backend: Backend,
                  image,
                  vector_field,
                  internal_dtype=cupy.float16,
                  block_size: int = 8):

    xp = backend.get_xp_module()
    source = r'''
                extern "C"{
                __global__ void warp_3d(float* warped_image,
                                        cudaTextureObject_t input_image,
                                        cudaTextureObject_t vector_field,
                                        int width, 
                                        int height,
                                        int depth)
                {
                    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
                    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
                    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

                    if (x < width && y < height && z < depth)
                    {
                        // coordinates in coord-normalised vector_field texture:
                        float u = float(x)/width;
                        float v = float(y)/height;
                        float w = float(z)/depth;

                        // Obtain linearly interpolated vector at (u,v,w):
                        float4 vector = tex3D<float4>(vector_field, u, v, w);

                        //printf("(%f, %f)=%f\n", sx, sy,  value);

                        // Obtain the shifted coordinates of the source voxel: 
                        float sx = float(x) + vector.x;
                        float sy = float(y) + vector.y;
                        float sz = float(z) + vector.z;

                        // Sample source image for voxel value:
                        float value = tex3D<float>(input_image, sx, sy, sz);

                        // Store interpolated value:
                        warped_image[z*width*depth + y*width + x] = value;

                        //TODO: supersampling would help in regions for which warping misses voxels in the source image,
                        //better: adaptive supersampling would automatically use the vector field divergence to determine where
                        //to super sample and by how much.  
                    }
                }
                }
                '''

    if image.ndim != 3 or vector_field.ndim != 4:
        raise ValueError("image or vector field has wrong number of dimensions!")

    # set up textures:
    input_image_tex = create_cuda_texture(image,
                                          num_channels=1,
                                          normalised_coords=False,
                                          sampling_mode='linear')

    vector_field = cupy.pad(vector_field, pad_width=((0,0),)*3+((0,1),), mode='constant')

    vector_field_tex = create_cuda_texture(vector_field,
                                           num_channels=4,
                                           normalised_coords=True,
                                           sampling_mode='linear')

    # Set up resulting image:
    warped_image = xp.empty_like(image)

    # get the kernel, which copies from texture memory
    warp_3d_kernel = cupy.RawKernel(source, 'warp_3d')

    # launch kernel
    width, height, depth = image.shape

    grid_x = (width + block_size - 1) // block_size
    grid_y = (height + block_size - 1) // block_size
    grid_z = (depth + block_size - 1) // block_size
    warp_3d_kernel((grid_x, grid_y, grid_z),
                   (block_size,) * 3,
                   (warped_image, input_image_tex, vector_field_tex, width, height, depth))

    return warped_image
