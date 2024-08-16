# Image-Plane Correction

This is a repository containing code/algorithms for image-plane correction, intended for use with the OVRO-LWA radio telescope.
This research is being performed by Zachary Huang as part of Caltech's 2024 SURF program (under the mentorship of Casey Law, Gregg Hallinan, and others).

# Libraries
We use [JAX](https://github.com/google/jax) to get easy GPU acceleration for matrix operations.
[Numpy](https://github.com/numpy/numpy) is still used in various places since some libraries don't support JAX arrays as a drop-in replacement.
We use [Astropy](https://github.com/astropy/astropy) for astronomical operations and [PyBDSF](https://github.com/lofar-astron/PyBDSF) for source detection.
[Matplotlib](https://github.com/matplotlib/matplotlib) is used for plotting.

# Development

All of the project dependencies are specified in `pyproject.toml`.
To install all of these dependencies along with the source code, enter the project directory and run `pip install .`.
If you wish to run the notebooks in `notebooks/`, I would recommend using Jupyter Lab (which you can install along with the project dependencies with `pip install .[dev]`.
For the sake of development, I would also recommend adding the `-e` flag to the pip commands above so that any changes made to the source code in this repository is immediately reflected by the scripts/notebooks (instead of requiring a "reinstall" of the package).
