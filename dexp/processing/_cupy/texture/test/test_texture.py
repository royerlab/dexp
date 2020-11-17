import cupy as cp

from dexp.processing._cupy.texture.texture import create_cuda_texture


def basic_cupy_texture_test():
    try:
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

        texobj, arr2 = create_cuda_texture(shape=(width, height), num_channels=1,  )

        # allocate input/output arrays
        tex_data = cp.arange(width * height, dtype=cp.float32).reshape(height, width)
        real_output = cp.zeros_like(tex_data)
        expected_output = cp.zeros_like(tex_data)
        arr2.copy_from(tex_data)
        arr2.copy_to(expected_output)

        # get the kernel, which copies from texture memory
        ker = cp.RawKernel(source, 'copyKernel')

        # launch it
        block_x = 4
        block_y = 4
        grid_x = (width + block_x - 1) // block_x
        grid_y = (height + block_y - 1) // block_y
        ker((grid_x, grid_y), (block_x, block_y), (real_output, texobj, width, height))

        # test outcome
        assert cp.allclose(real_output, expected_output)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")



def basic_cupy_texture_test():
    try:
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
        ch = cp.cuda.texture.ChannelFormatDescriptor(32, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
        arr2 = cp.cuda.texture.CUDAarray(ch, width, height)
        res = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=arr2)
        tex = cp.cuda.texture.TextureDescriptor((cp.cuda.runtime.cudaAddressModeClamp, cp.cuda.runtime.cudaAddressModeClamp),
                                                cp.cuda.runtime.cudaFilterModePoint,
                                                cp.cuda.runtime.cudaReadModeElementType)
        texobj = cp.cuda.texture.TextureObject(res, tex)

        # allocate input/output arrays
        tex_data = cp.arange(width * height, dtype=cp.float32).reshape(height, width)
        real_output = cp.zeros_like(tex_data)
        expected_output = cp.zeros_like(tex_data)
        arr2.copy_from(tex_data)
        arr2.copy_to(expected_output)

        # get the kernel, which copies from texture memory
        ker = cp.RawKernel(source, 'copyKernel')

        # launch it
        block_x = 4
        block_y = 4
        grid_x = (width + block_x - 1) // block_x
        grid_y = (height + block_y - 1) // block_y
        ker((grid_x, grid_y), (block_x, block_y), (real_output, texobj, width, height))

        # test outcome
        assert cp.allclose(real_output, expected_output)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")

