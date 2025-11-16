---
layout: page
title: CS180 - Project 4. Neural Radiance Fields
description: Rohan Shenoy
importance: 5
category: CS180
related_publications: false
---

In this project, we build an end-to-end nerf pipeline that is capable of reconstructing a 3D view of an object from a series of 2D pictures.

---

## Part 0: Calibration, Scanning, and Dataset Packaging

We start the projectby calibrating the camera, as different camera models effect our transformation to a pinhole-camera space. We begin by printing out ArUco tags and capturing a series of pictures of the tag from different angles, distances, and rotations ("pose"). We then feed the identified tags into 'cv2.calibrateCamera' to recover the camera intrisics and distortion affect.

Then, after we have the camera intrinsics, we capture our dataset of a singular ArUco tag by an object of our choice. We use a replica of the Cal Memorial Stadium. Here we try and keep the distance from the object the same, but vary the pose. It is better to stay close to the object (5-15 cm), as the error scales with the distance from the object.

To get our data ready for training, we need to calculate the view ("pose") of each image. Essentially, we want to understand where the camera is (rotation and translation of the camera), which is known as the extrinsic parameters of the camera. This is canonically known as the Perspective-n-Point (PnP) problem. After calculating this, we undistory our images and package the training data up for later. It is important to downsample here before packaging to speed up training.

Below we visualize our calculations of where the pictures are taken with respect to the object.

<div class="row">
  <div class="col-sm mt-2 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/Screenshot 2025-11-13 at 4.24.48 PM.png" title="Viser Frustum Visualization" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-2 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/Screenshot 2025-11-13 at 4.25.11 PM.png" title="Viser Frustum Visualization" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center">
  Visualizing the calibrated camera poses with Viser confirms consistent poses before NeRF training.
</div>

---

## Part 1: Neural Field for a 2D Image

Before fitting a 3D NeRF to our data, We will begin with fitting a 2D NeRF to an image. Essentially, this is just a simple Neural Field that can represent an image. We apply a positional encoding to the data before passing it through the linear layers of the neural network.

The neural network architecture is shown below. We use a learning rate of 1e-2.

<div class="row align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/Screenshot 2025-11-15 at 10.46.46 PM.png" title="Architecture Diagram" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center">
  Diagram of the neural field architecture showing PE inputs, the layered MLP with skip connections, and the Sigmoid/ReLU outputs for per-pixel color and density.
</div>


<div class="row align-items-center">
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_cat/reconstruction_iter0001.png" title="Iteration 1" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_cat/reconstruction_iter0015.png" title="Iteration 15" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_cat/reconstruction_iter0050.png" title="Iteration 50" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_cat/reconstruction_iter0200.png" title="Iteration 200" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_cat/reconstruction_iter2000.png" title="Iteration 2000" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center">
  Training progression on the cat image. Taken at iterations 1, 15, 50, 200 and 2000
</div>

<div class="row align-items-center">
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_emir/reconstruction_iter0001.png.png" title="Iteration 1" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_emir/reconstruction_iter0200.png" title="Iteration 200" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_emir/reconstruction_iter0400.png" title="Iteration 400" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_emir/reconstruction_iter1000.png.png" title="Iteration 1000" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_cat/reconstruction_iter2000.png" title="Iteration 2000" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center">
  Training progression on the Emir image. Taken at iterations 1, 200, 400, 1000 and 2000 
</div>

Here is the PSNR loss curve from the training on the Cat image.

<div class="row align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_cat/psnr_curve.png" title="Architecture Diagram" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

Here is a hyperparameter sweep that shows how the network size and the max positional encoding frequency affect the final results.

<div class="row align-items-center">
  <div class="col-sm mt-1 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_hyperparam/final_recon_grid.png" title="Architecture Diagram" class="img-fluid rounded z-depth-1" %}
  </div>
</div>




---

## Part 2: Multi-view Neural Radiance Field on Lego

With the Lego dataset (`lego_200x200.npz`), we parse intrinsics/focal, split images and poses, and build the ray sampler:

1. **Coordinate transforms:** Implement `transform(c2w, x_c)` and `pixel_to_camera(K, uv, s)` supporting batched points.
2. **Ray generation:** Use `pixel_to_ray(K, c2w, uv)` to produce origins/directions for every pixel, offsetting UVs by 0.5 to hit pixel centers.
3. **Sampling:** Draw N rays per iteration (either by image batching or global flattening), sample 32–64 points uniformly between near=2.0 and far=6.0, and jitter samples for training-time perturbations.
4. **Data loader:** Return `(ray_o, ray_d, pixel_color)` tuples, verify uv-ordering against stored images, and visualize rays/samples in Viser.

The network now accepts encoded 3D positions and view directions, injects the positional encodings midway through the MLP, outputs densities (ReLU) and colors (Sigmoid), and the volume rendering implementation accumulates contributions via the discrete integral. Training aims for >23 PSNR on validation using Adam with LR ≈ 5e-4 and sampling 10k rays per step. Deliverables cover textual descriptions of each part, ray/sample visualization (≤100 rays), training progression renders, validation PSNR curve, and a spherical video rendered from the provided test cameras.

<div class="row">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4/vis/ray_samples.png" title="Ray/Sample Visualization" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center">
  Rays sampled from a single training image and their perturbed sample points plotted in 3D confirm the frustra stay within the camera bounds.
</div>

---

## Part 2.6: Training on Personal Object Data

Finally, the undistorted, pose-labeled dataset created in Part 0 is used to train a NeRF of your own object. Adjust near/far bounds (e.g., 0.02–0.5) and increase samples per ray (64) when necessary, especially for handheld scans. Save intermediate renders and loss curves, then export a GIF of a camera circling the learned scene using scripts that rotate a camera pose around the origin. Discuss any custom hyperparameters or code changes applied during this stage. Deliverables include the loss-over-iterations plot, intermediate render snapshots, a novel-view GIF, and commentary on any tuning adjustments.

---

## Bells & Whistles (Optional)

- Depth map video for the Lego scene (CS 280A requirement)
- Advanced sampling (coarse-to-fine PDF resampling)
- Improved representations (e.g., Instant-NGP) with citations
- Novel background colors in the volume renderer
- Scene contraction (Mip-NeRF 360)
- nerfstudio-generated videos

Ensure all required web-hosted screenshots, GIFs, and plots are linked from the page when available.
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
