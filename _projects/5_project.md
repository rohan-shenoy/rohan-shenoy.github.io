---
layout: page
title: CS180 - Project 4. Neural Radiance Fields
description: Rohan Shenoy
importance: 5
category: Computer Vision Projects
related_publications: true
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
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_emir/reconstruction_iter0001.png" title="Iteration 1" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_emir/reconstruction_iter0200.png" title="Iteration 200" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_emir/reconstruction_iter0400.png" title="Iteration 400" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_emir/reconstruction_iter1000.png" title="Iteration 1000" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_emir/reconstruction_iter2000.png" title="Iteration 2000" class="img-fluid rounded z-depth-1" %}
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
    {% include figure.liquid path="assets/proj4_deliverables/nerf_2d_training_hyperparam/final_recon_grid.png" title="Hyperparameter Sweep" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

---

## Part 2: Multi-view Neural Radiance Field on Lego

Now we want to actually train a 3D NeRF to reconstruct a view of the stadium from the Part 0 dataset. We first begin by using the camera to world (c2w) matrix to write a function that takes a camera point in and transforms it into world space. We implement it in a manner that supports batched coordinates to speed up training using torch. We also implement a function that transform coordinates from the pixel coordinate system to the camera coordinate system, also support batched inputs. To do this, we multiply by the inverse w2c and intrinsic matrix K for each respective transformation. Finally, we create a function that takens in a pixel coordinate and converts it to a ray (origin and normalized direction), which we do by passing it through the aforementioned functions to have world coordinates, which can be used to compute the corresponding ray.

We then build a sampling algorithm that samples M images from our training dataset, and then samples N/M rays from each image to get N rays in total. We then write a function to uniformly create some samples along the ray (64 samples taken on each ray), based on some interval [near, far]. Each sample contains the location in 3D world coordinates and the corresponding color. 

Using these sampling functions, we build a dataloader that randomly samples pixels from the images in our dataset as created before. An important note here is we have to restrict the rays that can be sampled to be from an image in our dataset, otherwise we will just train the NeRF to predict all black as it is "learning" from rays that we do not actually have in our dataset.

Here is the architecture of our 3D NeRF.

<div class="row align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/Screenshot 2025-11-15 at 11.28.23 PM.png" title="Architecture Diagram" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center">
  Diagram of the neural radiance field architecture.
</div>

Here is a visualization of the rays on the lego dataset. 

<div class="row align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/Screenshot 2025-11-15 at 11.48.54 AM.png" title="Architecture Diagram" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center">
  Diagram of the neural radiance field architecture.
</div>

The last thing we implement is a function the discretely computes the value of a pixel given points along a series of rays that intersect with a pixel. This is so we can render an image from a certain direction after training the NeRF. 

We start by training on the lego dataset, demonstrating the training process below.


<div class="row align-items-center">
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/train_view0_iter_0001.png" title="Iteration 1" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/train_view0_iter_0050.png" title="Iteration 50" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/train_view0_iter_0400.png" title="Iteration 400" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/train_view0_iter_1000.png" title="Iteration 1000" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/train_view0_iter_2000.png" title="Iteration 2000" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

Here is the validation PSNR. Ignore the title and legend.

<div class="row align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/psnr_curve.png" title="PSNR curve" class="img-fluid rounded z-depth-1" %}
  </div>
</div>



Finally, here is the 3d reconstruction on the lego dataset:

<div class="row align-items-center">
  <div class="col-sm mt-3 mt-md-3 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/nerf_spherical_render_z.gif" title="3D view" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

Same things for my stadium dataset. The only hyperparameter changes was changing the near to 0.04 and the far to 0.8, computed by trial and error:

<div class="row align-items-center">
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/1.png" title="Iteration 50" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/2.png" title="Iteration 400" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/3.png" title="Iteration 1000" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/4.png" title="Iteration 2000" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-5 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/5.png" title="Iteration 5000" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="row align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/my_data_psnr.png" title="PSNR curve" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="row align-items-center">
  <div class="col-sm mt-3 mt-md-3 text-center">
    {% include figure.liquid path="assets/proj4_deliverables/orbit_iter_5000 (1).gif" title="3D view" class="img-fluid rounded z-depth-1" %}
  </div>
</div>




</div>
