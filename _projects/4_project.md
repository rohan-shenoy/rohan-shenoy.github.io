---
layout: page
title: Image Warping and Mosaicing
description: Rohan Shenoy
importance: 4
category: CS180
related_publications: false
---

This project explores image warping and mosaicing techniques to create seamless panoramic images. I implement homography computation, image warping with different interpolation methods, and blending techniques to combine multiple photographs into cohesive mosaics.

---

## Part A.1: Shoot the Pictures

I captured multiple sets of images with projective transformations between them by fixing the center of projection and rotating the camera. The images have significant overlap (40-70%) to facilitate registration.

### Image Set 1: Lecture

<div class="row">
  <div class="col-sm mt-2 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_1/rlecture1.jpeg" title="Lecture Image 1" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-2 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_1/rlecture2.jpeg" title="Lecture Image 2" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  First image set of cs180 lecture.
</div>

### Image Set 2: 7/11 + Car Crash

<div class="row">
  <div class="col-sm mt-2 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_3/r1.jpg" title="7/11 Image 1" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-2 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_3/r2.jpg" title="7/11 Image 2" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Second image set of 7/11 view from my balcony. You can notice a car being towed away in the right image after a car crash occured.
</div>

### Image Set 3: Campus

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_2/r1.jpg" title="Campus Image 1" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_2/r2.jpg" title="Campus Image 2" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_2/r3.jpg" title="Campus Image 2" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  3 images of campus that I will stitch to make a larger mosaic. Note that these were taken before the project was fully explained so there is some translation between the images rather than pure rotation. You will see the affect of this later on.
</div>

---

## Part A.2: Recover Homographies

I implemented a function to compute homographies from point correspondences using least-squares optimization.

### Theory

A homography H is a 3x3 matrix that relates corresponding points between two images: **p' = Hp**. With 8 degrees of freedom, we need at least 4 point correspondences to solve for H, though more points provide better stability. Typically I try to use 12-20, and then compute the least squares solution as the system is now overdetermined and has infinitely many solutions.

### Point Correspondences and Homography Results

Below we show the images with the correspondence points plotted on them. Below each set of images, we have the homography matrices computed for each one.

#### Lecture Images

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_1/rlecture1_with_points.png" title="Lecture 1 Correspondences Plotted" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_1/rlecture2_with_points.png" title="Lecture 2 Correspondences Plotted" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Point correspondences marked on lecture images.
</div>

**Homography Matrix:**
```
2.54224189,-0.02326480,-4665.08583807
0.58036296,2.01332082,-1374.72424197
0.00040159,-0.00003625,1.00000000
```

#### 7/11 Images

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_3/r1_with_points.png" title="7/11 Image 1 Correspondences Plotted" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_3/r2_with_points.png" title="7/11 Image 2 Correspondences Plotted" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Point correspondences marked on 7/11 images.
</div>

**Homography Matrix:**
```
3.39946906,-0.23491370,-6928.81047850
1.08393872,2.72711418,-2939.12346025
0.00061393,-0.00003907,1.00000000
```

#### Campus Images

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_2/r1_with_points.png" title="Campus Image 1 Correspondences" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_2/r2_with_points.png" title="Campus Image 2 Correspondences" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_2/r3_with_points.png" title="Campus Image 3 Correspondences" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Point correspondences marked on campus images.
</div>

**Homography Matrix 1 (Image 1 to Image 2):**
```
1.45918618,-0.04536459,-1514.89683294
0.18613671,1.22894351,-338.71987586
0.00012040,-0.00003097,1.00000000
```

**Homography Matrix 2 (Image 2 to Image 3):**
```
0.36786312,-0.14242818,91.61102915
-0.22004851,0.45603987,574.54088030
-0.00013562,-0.00012392,1.00000000
```

---

## Part A.3: Warp the Images

I implemented two interpolation methods for image warping using inverse warping to avoid holes in the output.

### Interpolation Methods

#### Nearest Neighbor Interpolation
- Rounds coordinates to the nearest pixel value
- Fast but can produce blocky artifacts
- Good for preserving sharp edges

#### Bilinear Interpolation  
- Uses weighted average of four neighboring pixels
- Smoother results but more computationally expensive
- Better for natural image content



### Rectification Results

Before creating mosaics, I tested the warping implementation by rectifying images containing rectangular objects.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/homographies/muir_woods.jpg" title="Original Muir Woods" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/homographies/muir_woods_warped_nn.jpg" title="NN Warped" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/homographies/muir_woods_warped_bilinear.jpg" title="Bilinear Warped" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Original image of Muir Woods map to save me from getting lost. Middle: Image rectified using nearest-neighbors. Right: Image rectified using bilinear weighting. 
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/homographies/van_gogh.jpg" title="Nearest Neighbor" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/homographies/van_gogh_warped_nn.jpg" title="NN Warped" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/homographies/van_gogh_warped_bilinear.jpg" title="Bilinear warped" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
    Left: Original image of Van Gogh painting. Middle: Van Gogh rectified using nearest-neighbors. Right: Van Gogh rectified using bilinear weighting. 
</div>

**Trade-offs:**
- Nearest neighbor is faster 
-For these images, the methods ended up doing the same transformation. Moving forward, I use the bilinear warping as I feel like it is the safer method in general to use.

---

## Part A.4: Blend the Images into a Mosaic

I created seamless mosaics by warping images into alignment and using blending to reduce edge artifacts.

### Blending Strategy

Instead of simple overlay which creates harsh edges, I implemented weighted averaging with alpha masks. Here's how the blending process works:

**Why Blending is Necessary:**
When you simply overlay warped images, you get harsh transitions and visible seams where images meet. This happens because:
- Small alignment errors create double images or gaps
- Different exposures between photos create brightness discontinuities  
- Camera settings may vary slightly between shots
-Scene changes quickly

**How to Use Blending:**
1. **Create Alpha Masks**: For each image, generate a mask where values are 1.0 at the center and gradually decrease to 0.0 at the edges
2. **Normalize Weights**: At each pixel, ensure all alpha values sum to 1.0 across all contributing images
3. **Weighted Average**: Compute the final pixel value as: `final_pixel = Σ(alpha_i × pixel_i) / Σ(alpha_i)`

**Alpha Mask Design:**
- **Linear Falloff**: Alpha decreases linearly from center to edge over a feathering distance
- **Distance Transform**: Use distance from image border to create smooth transitions
- **Overlap Handling**: In regions where multiple images overlap, blend based on relative distances from each image's center

**Benefits of This Approach:**
- Eliminates hard edges between images
- Reduces ghosting from small misalignments
- Handles exposure differences gracefully
- Creates natural-looking transitions

### Mosaic Results

#### Mosaic 1: Lecture

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_1/rlecture1_warped.jpg" title="Lecture 1 warped" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_1/rlecture2_warped.jpg" title="Lecture 2 warped" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_1/rlecture1_lecture2_mosaic.jpg" title="Final Mosaic" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption">
  First mosaic: Source images (left, middle) and final blended result (right). Notice that we get some blurriness due to people in the class moving between when the shots were taken.
</div>

#### Mosaic 2: 7/11 + Car Crash

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_3/r1_warped.jpg" title="7/11 Warped 1" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_3/r2_warped.jpg" title="7/11 Warped 2" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_3/r_mosaic.jpg" title="Final Mosaic" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Second mosaic showing the effectiveness of weighted blending for seamless stitching. This one is the best out of the 3.
</div>

#### Mosaic 3: Campus

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_2/r1_warped_to_r2_center.jpg" title="Campus Image 1" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_2/r2_center_reference.jpg" title="Campus Image 2" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_2/r3_warped_to_r2_center.jpg" title="Campus Image 3" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/Mosaic_2/r123_panorama_r2_ref.jpg" title="Final Mosaic" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
  </div>
</div>
<div class="caption">
  Third mosaic: Three source images combined into a wide panoramic view. You can see the affect of the aforementioned translation when taking the pictures. It makes the middle parts very blurry. Going forward, I know how to best rotate my phone to prevent any translation (like the technique used in the second set of images).
</div>

---

## Part B: Feature Matching for Autostitching

In this section, I implement the Multi-Image Matching algorithm based on the paper "Multi-Image Matching using Multi-Scale Oriented Patches" by Brown et al., with several simplifications. The goal is to automatically detect, match, and stitch images together without manual point selection as in the previous section.

### B.1: Harris Corner Detection

The first step involves detecting interest points using the Harris Corner Detector. Harris corners are points in an image where the intensity changes significantly in multiple directions, making them good candidates for matching across different images.

I used the provided Harris corner detection implementation to identify potential interest points. The algorithm computes the Harris response function at each pixel and identifies local maxima as corner candidates.

After detecting Harris corners, I implemented Adaptive Non-Maximal Suppression (ANMS) to select a more evenly distributed subset of the strongest corners. ANMS works by suppressing corners that are not significantly stronger than their neighbors within a certain radius, resulting in a more spatially distributed set of features.

#### Harris Corners Without ANMS
<div class="image-row">
    <div class="image-container">
        <img src="assets/proj3/results/lemon_trees/01_IMG_3387_corners_no_anms.png" alt="Harris corners image 1">
        <p class="caption">Left side of lemon trees Harris corners without ANMS</p>
    </div>
    <div class="image-container">
        <img src="assets/proj3/results/lemon_trees/02_IMG_3389_corners_no_anms.png" alt="Harris corners image 2">
        <p class="caption">Center of lemon trees Harris corners without ANMS</p>
    </div>
    <div class="image-container">
        <img src="assets/proj3/results/lemon_trees/03_IMG_3390_corners_no_anms.png" alt="Harris corners image 3">
        <p class="caption">Right side of lemon trees Harris corners without ANMS</p>
    </div>
</div>

#### Harris Corners With ANMS
<div class="image-row">
    <div class="image-container">
        <img src="assets/proj3/results/lemon_trees/01_IMG_3387_corners_anms.png" alt="Harris corners with ANMS image 1">
        <p class="caption">Left side of lemon trees Harris corners with ANMS</p>
    </div>
    <div class="image-container">
        <img src="assets/proj3/results/lemon_trees/02_IMG_3389_corners_anms.png" alt="Harris corners with ANMS image 2">
        <p class="caption">Center of lemon trees Harris corners with ANMS</p>
    </div>
    <div class="image-container">
        <img src="assets/proj3/results/lemon_trees/03_IMG_3390_corners_anms.png" alt="Harris corners with ANMS image 3">
        <p class="caption">Right side of lemon trees Harris corners with ANMS</p>
    </div>
</div>

We can see how ANMS spreads out the features that are selected throughout the image. This is important when trying to compute robust homographies.

### B.2: Feature Descriptor Extraction

Once interest points are detected, the next step is to extract distinctive feature descriptors around each corner. These descriptors capture the local appearance of the image region and enable matching between different images.

For each detected corner, I extract an 8x8 patch from a larger 40x40 window around the interest point. The larger window is first blurred to reduce noise and then sampled down to create the 8x8 descriptor. This approach provides better stability and distinctiveness compared to directly sampling the 8x8 patch.

The extracted descriptors are then normalized using bias and gain normalization to make them invariant to illumination changes. Bias normalization removes the mean, and gain normalization scales the descriptor to unit variance. This preprocessing step is crucial for robust matching across images with different lighting conditions.

### B.3: Feature Matching

Feature matching involves finding pairs of descriptors from different images that are likely to correspond to the same physical point in the scene. I use the nearest neighbor distance ratio test proposed by Lowe to identify good matches.

For each descriptor in the first image, I compute its distance to all descriptors in the second image. A match is considered valid if the ratio of the distance to the nearest neighbor and the distance to the second nearest neighbor is below a threshold (typically 0.7-0.8). This ratio test helps reject ambiguous matches where multiple descriptors are similarly close. Below I show some example features that are extracted and match between the 7/11 image (we show the entire 40x40 patch so it is easier to visualize how the matching process occurs, but for the actual computations, we use the normalized 8x8 patches).

#### Feature Matching Results
<div class="image-row">
    <div class="image-container">
        <img src="assets/proj3/results/7_11_crash/feature_patches_match_01.png" alt="Feature matches between image 1 and 2">
    </div>
    <div class="image-container">
        <img src="assets/proj3/results/7_11_crash/feature_patches_match_02.png" alt="Feature matches between image 2 and 3">
    </div>
</div>

<div class="image-row">
    <div class="image-container">
        <img src="assets/proj3/results/7_11_crash/feature_patches_match_03.png" alt="Feature matches between image 3 and 4">
    </div>
    <div class="image-container">
        <img src="assets/proj3/results/7_11_crash/feature_patches_match_04.png" alt="Feature matches between image 4 and 5">
    </div>
</div>

### B.4: RANSAC for Robust Homography

The final step involves computing a robust homography transformation using RANSAC (Random Sample Consensus). Even with the ratio test, some feature matches will be incorrect (outliers). RANSAC helps identify the largest set of consistent matches (inliers) and computes the homography from these reliable correspondences.

The 4-point RANSAC algorithm works by repeatedly:
1. Randomly selecting 4 point correspondences
2. Computing a homography from these 4 points
3. Evaluating how many other correspondences are consistent with this homography (within a small error threshold)
4. Keeping track of the homography with the most inliers

After many iterations, the homography with the most inliers is selected as the final transformation. This approach is robust to outliers and can handle situations where up to 50% of the matches are incorrect.

#### RANSAC Inlier Detection
<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/results/clark_kerr/left_matches.png" title="Left Image Matching" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/results/clark_kerr/center_matches.png" title=Center Image Matching" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj3/results/clark_kerr/right_matches.png" title="Right Image Matching" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  The matching points plotted on the clark kerr images.
</div>

#### Manual vs Automatic Stitching Comparison

The automatic feature-based approach produces different results compared to manual correspondence selection. While manual stitching allows for precise control over the alignment, automatic methods can sometimes capture more subtle geometric relationships. The automatic matching process is more robust when implemented correctly and is several orders of magnitudes faster (~30 seconds vs 10+ minutes). The 7/11 manual stitching is compared to the automatic cropping below:

<div class="image-row">
    <div class="image-container">
        <img src="assets/proj3/Mosaic_3/r_mosaic.jpg" alt="Manual stitching result">
        <p class="caption">Manual stitching result</p>
    </div>
    <div class="image-container">
        <img src="assets/proj3/results/7_11_crash/mosaic.png" alt="Automatic stitching result">
        <p class="caption">Automatic stitching result</p>
    </div>
</div>

### Automatic Mosaics Gallery

Here are some of the mosaics I created using this automatic stitching process. I make use of some clever techniques to produce some funny images.

<div class="mosaic-gallery">
    <div class="mosaic-row">
        <img src="assets/proj3/results/clark_kerr/mosaic.png" alt="Clark Kerr mosaic">
        <p class="caption">View from freshman year dorm</p>
    </div>
    <div class="mosaic-row">
        <img src="assets/proj3/results/home_run_mosaic/mosaic.png" alt="Home run">
        <p class="caption">Home run!</p>
    </div>
    <div class="mosaic-row">
        <img src="assets/proj3/results/lbnl_2/mosaic.png" alt="LBNL view">
        <p class="caption">View from Lawrence Berkeley National Labs</p>
    </div>
    <div class="mosaic-row">
        <img src="assets/proj3/results/lemon_trees/mosaic.png" alt="Clark Kerr Lemon Trees">
        <p class="caption">Lemon trees at Clark Kerr</p>
    </div>
    <div class="mosaic-row">
        <img src="assets/proj3/results/multi_cyp/mosaic.png" alt="Multi Cyp">
        <p class="caption">Multi Cyp</p>
    </div>
</div>
