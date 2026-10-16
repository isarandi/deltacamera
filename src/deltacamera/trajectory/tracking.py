"""PD-controller tracking camera with dead zone and inertia."""

import numpy as np



class TrackingCamera:
    """Causal PD-controller tracking camera.

    Maintains internal state (angular velocities, zoom velocity) and
    responds to per-frame target updates. The dead zone lets the subject
    drift within a region before the camera reacts, mimicking a human
    camera operator.

    Works in normalized image coordinates (resolution-independent).
    The dt parameter ensures frame-rate independence.

    Args:
        initial_camera: starting Camera (must have image_shape set).
        dead_zone: fraction of half-frame where no correction occurs.
        stiffness: proportional gain (how fast the camera responds).
        damping: derivative gain (anticipation, reduces oscillation).
        inertia: velocity decay per frame (0 = instant stop, 0.99 = very
            sluggish).
        zoom_stiffness: proportional gain for zoom correction.
        zoom_inertia: velocity decay for zoom.
        target_framing: desired fraction of frame height occupied by the
            target's bounding extent.
        min_fov: minimum field of view in degrees.
        max_fov: maximum field of view in degrees.
    """

    def __init__(self, initial_camera, *, dead_zone=0.1, stiffness=0.05,
                 damping=0.03, inertia=0.9, zoom_stiffness=0.03,
                 zoom_inertia=0.95, target_framing=0.6, min_fov=15.0,
                 max_fov=120.0):
        self._cam = initial_camera
        self._dead_zone = dead_zone
        self._stiffness = stiffness
        self._damping = damping
        self._inertia = inertia
        self._zoom_stiffness = zoom_stiffness
        self._zoom_inertia = zoom_inertia
        self._target_framing = target_framing
        self._min_fov = min_fov
        self._max_fov = max_fov
        self._yaw_vel = 0.0
        self._pitch_vel = 0.0
        self._zoom_vel = 0.0
        self._prev_error = np.zeros(2, np.float64)

    def step(self, target_world, dt=None, target_extent=None):
        """Advance one frame given a new target position.

        Args:
            target_world: world-space position to track, shape (3,).
            dt: time step in seconds. If None, assumed 1/30.
            target_extent: world-space size of the target (e.g., person
                height in mm). If provided, zoom adjusts to maintain
                target_framing.

        Returns:
            Camera for this frame.
        """
        if dt is None:
            dt = 1.0 / 30
        imshape = self._cam.image_shape or (1080, 1920)
        h, w = imshape[:2]
        half = np.array([w / 2.0, h / 2.0])
        proj = self._cam.world_to_image(
            np.asarray(target_world, np.float32).reshape(1, 3))[0]
        error = (proj - half) / half
        error_mag = np.linalg.norm(error)
        if error_mag < self._dead_zone:
            error = np.zeros(2)
        else:
            error = error * (error_mag - self._dead_zone) / error_mag
        d_error = error - self._prev_error
        self._prev_error = error.copy()
        correction = self._stiffness * error + self._damping * d_error
        self._yaw_vel = self._inertia * self._yaw_vel + (1 - self._inertia) * correction[0]
        self._pitch_vel = self._inertia * self._pitch_vel + (1 - self._inertia) * correction[1]
        self._cam = self._cam.rotated(
            yaw=float(self._yaw_vel * dt),
            pitch=float(self._pitch_vel * dt))
        if target_extent is not None:
            cam_pts = self._cam.world_to_camera(
                np.asarray(target_world, np.float32).reshape(1, 3))[0]
            depth = max(float(cam_pts[2]), 100.0)
            fov_current = self._cam.get_fov(imshape)
            apparent_frac = float(target_extent) / (
                2 * depth * np.tan(np.radians(fov_current / 2)))
            zoom_error = apparent_frac - self._target_framing
            self._zoom_vel = (self._zoom_inertia * self._zoom_vel
                              + (1 - self._zoom_inertia)
                              * self._zoom_stiffness * zoom_error)
            zoom_factor = 1.0 + float(self._zoom_vel * dt)
            new_fov = fov_current / max(zoom_factor, 0.5)
            new_fov = np.clip(new_fov, self._min_fov, self._max_fov)
            self._cam = self._cam.zoomed(fov_current / new_fov)
        return self._cam

    def step_multi(self, target_worlds, dt=None):
        """Advance one frame given multiple target positions.

        Tracks the centroid; zoom adjusts to contain the bounding extent.
        """
        pts = np.asarray(target_worlds, np.float32)
        centroid = pts.mean(0)
        extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        return self.step(centroid, dt=dt, target_extent=extent)

    @property
    def camera(self):
        return self._cam

    def reset(self, camera=None):
        """Reset velocities. Optionally set a new camera."""
        self._yaw_vel = 0.0
        self._pitch_vel = 0.0
        self._zoom_vel = 0.0
        self._prev_error = np.zeros(2, np.float64)
        if camera is not None:
            self._cam = camera
