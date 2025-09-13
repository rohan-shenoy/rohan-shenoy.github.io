---
layout: page
title: CS180 - Project 1. The Prokudin-Gorskii Collection
description: Rohan Shenoy
importance: 2
category: CS180
related_publications: false
---

This project focuses on the digitized glass plate images from Sergei Prokudin-Gorskii, a pioneer of color photography. His technique involved capturing three monochrome images of a scene in quick succession, each through a different color filter (red, green, and blue). Our task is to take these separated channels and align them to reconstruct the original color photograph.

<div class="row align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/cathedral_uncolored.png" title="Unaligned Cathedral" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/emir_uncolored.png" title="Unaligned Emir" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center">
  The raw, uncolored glass plate negatives for the Cathedral and Emir images. Each contains the red, green, and blue channels stacked vertically.
</div>

---

### Naive Channel Overlay

The simplest approach to creating a color image is to split the three channels and directly stack them. However, because the plates were captured separately, they are almost always misaligned. This naive overlay results in significant color fringing and a blurry, distorted image, as seen below.

<div class="row align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/direct_cathedral.png" title="Direct Cathedral" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/direct_emir.png" title="Direct Emir" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center">
  Directly overlaying the R, G, and B channels reveals severe misalignment.
</div>

---

### Single-Scale Alignment

To fix the misalignment, we can use a single-scale alignment algorithm. This method performs an exhaustive search over a small pixel window, shifting one channel relative to a reference channel. The best alignment is the displacement that maximizes the similarity between the two channels, according to a chosen distance metric. We use two primary distance metrics:

1.  **L2 Norm (Sum of Squared Differences):** This metric calculates the direct Euclidean distance between the pixel values of two images. It's fast but sensitive to overall brightness differences.
    $$ L_2 = \sqrt{\sum_{i=1}^{n}(u_i - v_i)^2} $$
2.  **Normalized Cross-Correlation (NCC):** This metric is more robust to linear variations in brightness and contrast. It normalizes both images before comparing them.
    $$ NCC = \frac{1}{n} \sum_{i=1}^{n} \frac{(u_i - \bar{u})}{\sigma_u} \frac{(v_i - \bar{v})}{\sigma_v} $$

Below are the single-scale alignment results for the Cathedral image.

<div class="row align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/cathedral_l2.png" title="Cathedral L2" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/cathedral_ncc.png" title="Cathedral NCC" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center">
  Cathedral aligned with L2 Norm (left) and Normalized Cross-Correlation (right).
</div>

---

### Multi-Scale Pyramid Alignment

For high-resolution images like `emir.tif`, an exhaustive search is computationally expensive. A more efficient strategy is the multi-scale image pyramid. This approach creates a series of downsampled images (a pyramid). It first finds a coarse alignment on the smallest image, then refines the search in a small window at progressively higher resolutions. This coarse-to-fine strategy is significantly faster and more effective for large displacements.

<div class="row align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/emir.png" title="Emir Aligned" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center">
  The Emir image, aligned using the efficient multi-scale pyramid method.
</div>

---

### Alignment Gallery

Here are more results from aligning other images from the Prokudin-Gorskii collection, displayed one per row for better visibility.

<div class="row mt-3 align-items-center">
  <div class="col-sm text-center">
    {% include figure.liquid path="/assets/proj1_outputs/church.png" title="Church" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row mt-3 align-items-center">
  <div class="col-sm text-center">
    {% include figure.liquid path="/assets/proj1_outputs/harvesters.png" title="Harvesters" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row mt-3 align-items-center">
  <div class="col-sm text-center">
    {% include figure.liquid path="/assets/proj1_outputs/icon.png" title="Icon" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row mt-3 align-items-center">
  <div class="col-sm text-center">
    {% include figure.liquid path="/assets/proj1_outputs/melons.png" title="Melons" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row mt-3 align-items-center">
  <div class="col-sm text-center">
    {% include figure.liquid path="/assets/proj1_outputs/monastery.png" title="Monastery" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row mt-3 align-items-center">
  <div class="col-sm text-center">
    {% include figure.liquid path="/assets/proj1_outputs/self_portrait.png" title="Self Portrait" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row mt-3 align-items-center">
  <div class="col-sm text-center">
    {% include figure.liquid path="/assets/proj1_outputs/three_generations.png" title="Three Generations" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row mt-3 align-items-center">
  <div class="col-sm text-center">
    {% include figure.liquid path="/assets/proj1_outputs/tobolsk.png" title="Tobolsk" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row mt-3 align-items-center">
  <div class="col-sm text-center">
    {% include figure.liquid path="/assets/proj1_outputs/siren.png" title="Siren" class="img-fluid rounded z-depth-1" %}
  </div>
</div>


---

### Final Gallery

Here are the final, processed images, which have been aligned using the L2 Norm, enhanced with adaptive histogram equalization, and cropped by 5% to remove noisy borders. Histogram equalization is a contrast enhancement technique that redistributes the pixel intensities of an image to be as uniform as possible. It works by creating a new intensity mapping that spreads out the most frequent brightness values. The result is an image with a wider range of darks and lights, making features and textures more distinct and visible.

<div class="row mt-3 align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/outputs_final/church_final.png" title="Church" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/outputs_final/emir_final.png" title="Emir" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row mt-3 align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/outputs_final/harvesters_final.png" title="Harvesters" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/outputs_final/icon_final.png" title="Icon" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row mt-3 align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/outputs_final/italil_final.png" title="Italil" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/outputs_final/lastochikino_final.png" title="Lastochikino" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row mt-3 align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/outputs_final/lugano_final.png" title="Lugano" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/outputs_final/melons_final.png" title="Melons" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row mt-3 align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/outputs_final/self_portrait_final.png" title="Self Portrait" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/outputs_final/siren_final.png" title="Siren" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row mt-3 align-items-center">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.liquid path="/assets/proj1_outputs/outputs_final/three_generations_final.png" title="Three Generations" class="img-fluid rounded z-depth-1" %}
  </div>
</div>