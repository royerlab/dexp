from arbol import asection, aprint

from dexp.processing.backends._cupy.texture.texture import create_cuda_texture
from dexp.processing.backends.cupy_backend import CupyBackend


def test_cupy_texture_4channels():
    try:
        import cupy
        with CupyBackend():
            source = r'''
                extern "C"{
                __global__ void copyKernel(float* output,
                                           cudaTextureObject_t texObj,
                                           int width, int height)
                {
                    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
                    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
                    // Read from texture and write to global memory
                    float u = x+0.5f;
                    float v = y+0.5f;
                    if (x < width && y < height)
                    {
                        output[y * 4 *width + 4 *x +0] = tex2D<float4>(texObj, u, v).x;
                        output[y * 4 *width + 4 *x +1] = tex2D<float4>(texObj, u, v).y;
                        output[y * 4 *width + 4 *x +2] = tex2D<float4>(texObj, u, v).z;
                        output[y * 4 *width + 4 *x +3] = tex2D<float4>(texObj, u, v).w;
                    }
                }
                }
                '''
            width = 3
            height = 5

            # allocate input/output arrays
            tex_data = cupy.arange(width * height * 4, dtype=cupy.float32).reshape(height, width, 4)

            # set up a texture object
            texobj, cuda_array = create_cuda_texture(tex_data,
                                                     num_channels=4,
                                                     sampling_mode='nearest',
                                                     dtype=cupy.float32)

            real_output = cupy.zeros_like(tex_data)
            expected_output = tex_data.copy()

            # get the kernel, which copies from texture memory
            kernel = cupy.RawKernel(source, 'copyKernel')

            # launch it
            block_x = 4
            block_y = 4
            grid_x = (width + block_x - 1) // block_x
            grid_y = (height + block_y - 1) // block_y
            kernel((grid_x, grid_y), (block_x, block_y), (real_output, texobj, width, height))

            del texobj, cuda_array

            # test outcome
            assert cupy.allclose(real_output, expected_output)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def test_cupy_texture_1channel_normcoord():
    try:
        import cupy
        with CupyBackend():
            source = r'''
                
                extern "C"{
                __global__ void texture_1channel_normcoord_kernel(float* output,
                                           cudaTextureObject_t texObj,
                                           int width, 
                                           int height)
                {
                    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
                    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
                    // Read from texture and write to global memory
                    float u = (float(x)+0.5f)/width;
                    float v = (float(y)+0.5f)/height;
                    
                    if (x < width && y < height)
                    {
                        float value = tex2D<float>(texObj, u, v);
                        printf("(%f, %f)=%f\n", u, v, value);
                        output[y * width + x] = value;
                    }
                }
                }
                '''
            width = 3
            height = 5

            # allocate input/output arrays
            tex_data = cupy.arange(width * height, dtype=cupy.float32).reshape(height, width)

            # set up a texture object
            texobj, cuda_array = create_cuda_texture(tex_data,
                                                     num_channels=1,
                                                     normalised_coords=True,
                                                     sampling_mode='linear',
                                                     dtype=cupy.float32)

            real_output = cupy.zeros_like(tex_data)
            expected_output = tex_data.copy()

            # get the kernel, which copies from texture memory
            kernel = cupy.RawKernel(source, 'texture_1channel_normcoord_kernel')

            # launch it
            block_x = 4
            block_y = 4
            grid_x = (width + block_x - 1) // block_x
            grid_y = (height + block_y - 1) // block_y
            kernel((grid_x, grid_y), (block_x, block_y), (real_output, texobj, width, height))

            del texobj, cuda_array

            # test outcome
            assert cupy.allclose(real_output, expected_output)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def test_cupy_texture_1channel():
    try:
        import cupy
        with CupyBackend():
            source = r'''
                extern "C"{
                __global__ void copyKernel(float* output,
                                           cudaTextureObject_t texObj,
                                           int width, int height)
                {
                    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
                    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
                    // Read from texture and write to global memory
                    float u = x+0.5f;
                    float v = y+0.5f;
                    if (x < width && y < height)
                    {
                        float value = tex2D<float>(texObj, u, v);
                        printf("(%f, %f)=%f\n", u, v, value);
                        output[y * width + x] = value;
                    }
                }
                }
                '''
            width = 3
            height = 5

            # allocate input/output arrays
            tex_data = cupy.arange(width * height, dtype=cupy.float32).reshape(height, width)
            tex_data[1, 2] = 1

            # set up a texture object
            texobj, cuda_array = create_cuda_texture(tex_data,
                                                     num_channels=1,
                                                     sampling_mode='linear',
                                                     dtype=cupy.float32)

            real_output = cupy.zeros_like(tex_data)
            expected_output = tex_data.copy()

            # get the kernel, which copies from texture memory
            kernel = cupy.RawKernel(source, 'copyKernel')

            # launch it
            block_x = 4
            block_y = 4
            grid_x = (width + block_x - 1) // block_x
            grid_y = (height + block_y - 1) // block_y
            kernel((grid_x, grid_y), (block_x, block_y), (real_output, texobj, width, height))

            del texobj, cuda_array

            # test outcome
            assert cupy.allclose(real_output, expected_output)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def test_basic_cupy_texture():
    try:
        import cupy
        with CupyBackend():
            source = r'''
                extern "C"{
                __global__ void copyKernel(float* output,
                                           cudaTextureObject_t texObj,
                                           int width, int height)
                {
                    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
                    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
                    // Read from texture and write to global memory
                    float u = x;
                    float v = y;
                    if (x < width && y < height)
                        output[y * width + x] = tex2D<float>(texObj, u, v);
                }
                }
                '''
            width = 8
            height = 16

            # set up a texture object
            ch = cupy.cuda.texture.ChannelFormatDescriptor(32, 0, 0, 0, cupy.cuda.runtime.cudaChannelFormatKindFloat)
            arr2 = cupy.cuda.texture.CUDAarray(ch, width, height)
            res = cupy.cuda.texture.ResourceDescriptor(cupy.cuda.runtime.cudaResourceTypeArray, cuArr=arr2)
            tex = cupy.cuda.texture.TextureDescriptor((cupy.cuda.runtime.cudaAddressModeClamp,
                                                       cupy.cuda.runtime.cudaAddressModeClamp),
                                                      cupy.cuda.runtime.cudaFilterModePoint,
                                                      cupy.cuda.runtime.cudaReadModeElementType)
            texobj = cupy.cuda.texture.TextureObject(res, tex)

            # allocate input/output arrays
            tex_data = cupy.arange(width * height, dtype=cupy.float32).reshape(height, width)
            real_output = cupy.zeros_like(tex_data)
            expected_output = cupy.zeros_like(tex_data)
            arr2.copy_from(tex_data)
            arr2.copy_to(expected_output)

            # get the kernel, which copies from texture memory
            ker = cupy.RawKernel(source, 'copyKernel')

            # launch it
            block_x = 4
            block_y = 4
            grid_x = (width + block_x - 1) // block_x
            grid_y = (height + block_y - 1) // block_y
            ker((grid_x, grid_y), (block_x, block_y), (real_output, texobj, width, height))

            del texobj, arr2

            # test outcome
            assert cupy.allclose(real_output, expected_output)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def test_basic_cupy_texture_leak():
    try:
        import cupy
        with CupyBackend():
            # allocate input/output arrays
            length = 512
            tex_data = cupy.arange(length ** 3, dtype=cupy.float32).reshape(length, length, length)

            with asection("loop"):
                for i in range(100):
                    aprint(f"i={i}")
                    texobj, cuda_array = create_cuda_texture(tex_data,
                                                             num_channels=1,
                                                             sampling_mode='linear',
                                                             dtype=cupy.float32)

    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")
