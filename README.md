# ParticleFilterTracker

Particle filtering is one of the key algorithms in the field of probabilistic robotics. They are
specifcally useful in cases where the system's dynamic model and measurement functions
are non-linear and non-Gaussian. In addition, they have been shown to perform well in
cluttered scenes and in cases of short-duration occlusions. Particle filters are able to
represent such distributions by representing a distribution by a set of "particles", which are
a set of weighted samples.

The control system first needs to be fed with a target region to track. This can done by an
instance segmentation algorithm, but for this study the target region is manually specifed by drawing a bounding box using mouse input.
A stereo-depth camera attached to the user's AR glasses is used to capture an RGB-D image for each frame using an Intel RealSense D435i camera.
![alt text](https://github.com/Mena-SA-Kamel/ParticleFilterTracker/blob/master/Tracker_example.PNG)
![alt text](https://github.com/Mena-SA-Kamel/ParticleFilterTracker/blob/master/Tracker_example_motion_history.PNG)
