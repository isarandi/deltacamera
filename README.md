# Deltacamera

Represent, manipulate and use camera calibration info in computer vision tasks.

Main features:

- The library supports **converting coordinates** between world, camera and image space, handing **lens distortion** models according to the Brown–Conrady and Kannala–Brandt models.
- Modify cameras with intuitive methods such as `camera.zoom`, `camera.rotate`, `camera.scale_output`, `camera.turn_towards`, etc.
- Conversion between distorted and undistorted image spaces are also implemented in an efficient way using Numba and **a more accurate inversion of Brown–Conrady distortion** compared to OpenCV. We use Newton's method in addition to the standard fixed-point iteration. This library can also keep track of valid image regions after warping, inspired by [Leotta et al.](https://openaccess.thecvf.com/content/WACV2022/papers/Leotta_On_the_Maximum_Radius_of_Polynomial_Lens_Distortion_WACV_2022_paper.pdf), but extended to the full Brown-Conrady and Kannala-Brandt models.

- This library also includes efficient implementations of **image warping**, with antialiasing support and interpolation in linear RGB color space. The warping maps can be cached for very fast repeated use (e.g., warp/undistort a video taken from a static camera to another calibration setup). This also supports partial caching of only the more expensive distortion part. This is useful when the rotation can change during a video, but the distortion parameters are fixed (e.g., turning the camera to keep the subject centered).

## Installation

```bash
pip install deltacamera
```

It is recommended to then run the Numba precompilation step (takes around 1–2 minutes). This will make image warping and coordinate transformations fast already on first use.

```bash
python -m deltacamera.precompile
```

## Documentation

Full documentation is available at [deltacamera.readthedocs.io](https://deltacamera.readthedocs.io).

## References

For the idea of computing the valid image region after distortion, see:
- Matthew J. Leotta, David Russell, Andrew Matrai, "On the Maximum Radius of Polynomial Lens Distortion", WACV 2022.
