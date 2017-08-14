from distutils.core import setup

setup(
    name='kerasdl4j',
    version='0.1',
    description='Use Deeplearning4j as backend for Keras',
    author='Justin Long',
    author_email='help@skymind.ai',
    url='https://github.com/deeplearning4j/dl4j-examples',
    packages=['kerasdl4j'],
    install_requires=['keras', 'py4j', 'h5py', 'xxhash'],
)
