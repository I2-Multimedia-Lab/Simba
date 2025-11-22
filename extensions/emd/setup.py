from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd',
    ext_modules=[
        CUDAExtension(
            'emd',
            [
                'emd.cpp',
                'emd_cuda.cu',
            ],
            extra_compile_args={'cxx': ['-fno-strict-aliasing'], 'nvcc': []}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)