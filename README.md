# Image-Plane Correction

This is a repository containing code/algorithms for image-plane correction, intended for use with the OVRO-LWA radio telescope.
This research is being performed by Zachary Huang as part of Caltech's 2024 SURF program (under the mentorship of Casey Law, Gregg Hallinan, and others).

# Libraries
We use [JAX](https://github.com/google/jax) to get easy GPU acceleration for matrix operations.
[Numpy](https://github.com/numpy/numpy) is still used in various places since some libraries don't support JAX arrays as a drop-in replacement.
We use [Astropy](https://github.com/astropy/astropy) for astronomical operations and [PyBDSF](https://github.com/lofar-astron/PyBDSF) for source detection.
[Matplotlib](https://github.com/matplotlib/matplotlib) is used for plotting.

[OpenCV](https://github.com/opencv/opencv) ([with extra modules](https://github.com/opencv/opencv_contrib)) is used for optical flow.
However, note that for CUDA support, OpenCV must be built from scratch.
This can be a bit of a pain, so I wrote down the installation steps that worked for me in `docs/opencv.md`.

# Development

All of the project dependencies (except for OpenCV) are specified in `pyproject.toml`.
To install these dependencies along with the source code, enter the project directory and run `pip install .`.
If you wish to run the notebooks in `notebooks/`, I would recommend using Jupyter Lab (which you can install along with the project dependencies with `pip install .[dev]`.
For the sake of development, I would also recommend adding the `-e` flag to the pip commands above so that any changes made to the source code in this repository is immediately reflected by the scripts/notebooks (instead of requiring a "reinstall" of the package).

For GPU acceleration, ensure that JAX is installed with [CUDA support](https://jax.readthedocs.io/en/latest/installation.html#installation).
As mentioned previously OpenCV must be built from source for CUDA support and should be accessible as `cv2`.
