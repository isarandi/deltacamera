# DeltaCamera

This library is about camera models, coordinate transformations, and image warping/undistortion to account for changing camera parameters and lens distortion.

## Main use cases

- **Convert coordinates** between world, camera and image space, handing lens distortion models according to the 14-parameter extended Brown–Conrady and the 4-parameter Kannala–Brandt fisheye models.
- **Reproject** image points and entire images between different camera calibration settings (e.g., rotate the camera or change lens distortion).
- Modify cameras with **intuitive methods** such as `camera.zoomed`, `camera.rotated`, `camera.image_resized`, `camera.turned_towards`, etc.

## Benefits

- **Speed:** The critical functions are accelerated with Numba and intermediate computations are cached for high performance.
- **Accuracy:** We use a more accurate inversion of Brown–Conrady compared to OpenCV. OpenCV uses only fixed-point iteration, we use Newton's method in addition to that.
- **Tracking of pixel validity:** When warping images, we keep track of which pixels in the output image are valid (i.e., map to valid pixels in the input image), taking into account also the valid region of the lens distortion model (outside of which distortion folds back on itself, and would cause artifacts if not detected). The valid region can be obtained as a Shapely polygon, an RLEMaskLib mask or as NaNs in the return values. This feature is missing from OpenCV.
- **Linear color interpolation:** When warping images, we use linear interpolation in linear color space (gamma corrected), which avoids artifacts when warping images with strong contrast.
- **Anti-aliasing:** When warping images, we use supersampling to avoid aliasing artifacts when downsampling parts of the image.

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
