---
layout: page
title: CS180 - Project 0. Becoming Friends with Your Camera
description: Rohan Shenoy
importance: 1
category: CS180
related_publications: false
---

### Part 1: Selfie — The Wrong Way vs. The Right Way

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/Proj0/selfie-comparison/selfie1.jpg" title="image1" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/Proj0/selfie-comparison/selfie2.jpg" title="image2" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/Proj0/selfie-comparison/selfie3.jpg" title="image3" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/Proj0/selfie-comparison/selfie4.jpg" title="image4" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/Proj0/selfie-comparison/selfie5.jpg" title="image5" class="img-fluid rounded z-depth-1 fixed-thumb" %}
  </div>
</div>

<div class="caption text-center mt-2">
  Facial features become increasingly distorted as we move the camera closer to the subject and zoom out.
</div>

<p>
When you hold the camera close to the subject's face, the lens has to use a wide field of view to display the entire face. Wide angles overemphasize perspective, with features closer to the lens (i.e., a nose) looking disproportionately large, while features farther back (ears, background) look smaller. Since selfies are typically captured from up close, perspective is exaggerated, so the subject looks unnatural. At a greater distance and zooming in on the subject, you’re narrowing the field of view. The relative size difference between near and far points on the face and in the background shrinks, so the proportions come out more natural. In other words, you _approach_ an orthogonal projection of the field of view.
</p>

---

### Part 2: Architectural Perspective Compression

<div class="row">
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/Proj0/urban-perspective/urban0.jpg" title="arch1" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/Proj0/urban-perspective/urban1.jpg" title="arch2" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/Proj0/urban-perspective/urban2.jpg" title="arch3" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/Proj0/urban-perspective/urban3.jpg" title="arch4" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/Proj0/urban-perspective/urban4.jpg" title="arch5" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption text-center mt-2">
  Demonstration of perspective compression across two rows of images.
</div>

<p>
When you stand far away and zoom in down a street, the lens narrows its field of view. This compresses perspective: objects that are actually separated by large distances along the street appear stacked closely together, making the scene look flattened. Buildings, lampposts, and people at different depths seem almost the same size.
</p>

---

### Part 3: Bringing It All Together

<p>
When you move the camera backward while zooming in, you change the perspective but keep the subject about the same size. The background appears to stretch and warp around the subject, creating the dolly zoom effect.
</p>

<div class="text-center my-4">
  {% include figure.liquid path="assets/Proj0/dolly-zoom/I_want_you.gif" title="caught_lacking.gif" class="img-fluid rounded z-depth-1 mx-auto d-block" style="max-width: 50%;" %}
</div>

<div class="caption text-center mt-2">
  Just put the fries in the bag, bro
</div>

---

<!-- Custom CSS for thumbnails -->
<style>
.fixed-thumb img {
  height: 200px;
  object-fit: cover;
  width: 100%;
}
</style>