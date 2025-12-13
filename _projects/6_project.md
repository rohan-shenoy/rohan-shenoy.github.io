---
layout: page
title: Fun With Diffusion Models!
description: Rohan Shenoy
importance: 6
category: CS180
related_publications: false
---

This project explores the power of diffusion models, specifically focusing on Denoising Diffusion Probabilistic Models (DDPMs). The project is divided into two parts: first, experimenting with pre-trained diffusion models for image generation and editing, and second, implementing and training a diffusion model from scratch on the MNIST dataset.

You can view the detailed project specifications here:
- [Part A: The Power of Diffusion Models](https://cal-cs180.github.io/fa25/hw/proj5/parta.html)
- [Part B: Diffusion Models from Scratch](https://cal-cs180.github.io/fa25/hw/proj5/partb.html)

---

## Part A: The Power of Diffusion Models

In the first part, I deployed DeepFloyd IF, a text-to-image diffusion model, to perform various generation tasks. Key experiments included:

### Sampling Loops
Implementing the denoising loop manually to understand how noise is iteratively removed to form an image.

### Inpainting
Using a mask to guide the diffusion process, allowing for the replacement of specific parts of an image while keeping the rest consistent.

### Visual Anagrams & Hybrid Images
Creating optical illusions by denoising an image such that it looks like one prompt when upright and another when flipped, or combining low and high frequencies of two different prompts.

### ControlNet
Utilizing ControlNet to guide generation with structural constraints like Canny edges.

---

## Part B: Diffusion Models from Scratch

The second part involved building the architecture for a diffusion model and training it.

### UNet Architecture
Implemented a UNet-based noise predictor with time-conditioning to predict noise at various timesteps.

### Training on MNIST
Trained the model to generate handwritten digits by reversing the diffusion process.

### Class Conditioning
Extended the model to accept class labels, allowing for the targeted generation of specific digits (0-9).
You describe how you toiled, sweated, _bled_ for your project, and then... you reveal its glory in the next row of images.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>

The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above:

{% raw %}

```html
<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
```

{% endraw %}
