import ctypes
import os
from typing import Optional


def set_mkl_threads(num_threads: Optional[int] = None) -> None:
    try:
        if num_threads is None:
            import multiprocessing

            num_threads = multiprocessing.cpu_count() // 2 - 1

        try:
            import mkl

            mkl.set_num_threads(num_threads)
            return 0

        except Exception:
            pass
            # import traceback
            # traceback.print_exc()
            # print(e)

        for name in ["libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]:
            try:
                mkl_rt = ctypes.CDLL(name)
                mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(num_threads)))
                return 0

            except Exception:
                pass
                # import traceback
                # traceback.print_exc()
                # print(e)

        os.environ["OMP_NUM_THREADS"] = f"{num_threads}"  # export OMP_NUM_THREADS=4
        os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"  # export OPENBLAS_NUM_THREADS=4
        os.environ["MKL_NUM_THREADS"] = f"{num_threads}"  # export MKL_NUM_THREADS=6
        os.environ["VECLIB_MAXIMUM_THREADS"] = f"{num_threads}"  # export VECLIB_MAXIMUM_THREADS=4
        os.environ["NUMEXPR_NUM_THREADS"] = f"{num_threads}"  # export NUMEXPR_NUM_THREADS=6

    except Exception:
        pass
        # import traceback
        # traceback.print_exc()
        # print(e)


# set_mkl_threads(8)
