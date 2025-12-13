---
layout: page
title: Fun With Diffusion Models!
description: Rohan Shenoy
importance: 6
category: CS180
related_publications: false
---

This project explores the power of diffusion models. The project is divided into two parts:  
**Part A** investigates the capabilities of large pretrained diffusion models for image generation and editing, while in **Part B** we develop and train our own diffusion model on the MNIST dataset.

---

## Part A: The Power of Diffusion Models

In Part A, I use the pretrained **DeepFloyd IF**, a large text-to-image diffusion model, to experiment with diffusion.

### A.0 Playing with DeepFloyd IF
To begin, I wrote some prompts out and that used a prompt encoder to generate the respective prompt embeddings. We then pass these prompt embeddings through the model to see the outputs. For the below prompts, we use "boots filled with flower", "a chair growing leaves", and "beer smoking a cigarette". We set num_inference to 10 (top row) and 50 (bottom row). All randomization was seeded with seed 42.


<div class="row justify-content-sm-center">
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/0/boots_flowers_10.png" title="boots filled with flower" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Prompt: “boots filled with flowers”
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/0/leafy_chair_10.png" title="a chair growing leaves" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Prompt: “a chair growing leaves”
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/0/beer_cig_10.png" title="beer smoking a cigarette" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Prompt: “beer smoking a cigarette”
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/0/boots_flowers_50.png" title="boots filled with flower" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Prompt: “boots filled with flower”
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/0/leafy_chair_50.png" title="a chair growing leaves" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Prompt: “a chair growing leaves”
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/0/beer_cig_50.png" title="beer smoking a cigarette" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Prompt: “beer smoking a cigarette”
    </div>
  </div>
</div>

---

### A.1.1 Forward Process

A core component of diffusion models is the **forward diffusion process**, which gradually adds noise to a clean image where the noise is Gaussian distributed. Formally, the forward process defines a conditional distribution

\[
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\ (1-\bar{\alpha}_t)I\right),
\]

which can be equivalently written as

\[
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0, I).
\]

Here, \(x_0\) is the original clean image, \(x_t\) is the noisy image at timestep \(t\), and \(\bar{\alpha}_t\) is a precomputed noise schedule that decreases monotonically with \(t\). As \(t\) increases, the signal term \(\sqrt{\bar{\alpha}_t}x_0\) diminishes while the noise term dominates, causing the image to become progressively more corrupted. I implement this forward process to visualize how information is destroyed over time and to generate training data for the reverse denoising model, which learns to invert this corruption process by predicting clean structure from noisy inputs. We visualize this process below on an image of the Berkeley Campanile.

<div class="row justify-content-sm-center">
  <div class="col-sm-4 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_2/1.7.2campinelle_original.png" title="Original image" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Original image
    </div>
  </div>
  <div class="col-sm-4 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_1/1.1_noisy_250.png" title="Campanile after 250 steps of noise" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile after 250 steps of noising
    </div>
  </div>
  <div class="col-sm-4 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_1/1.1_noisy_500.png" title="Campanile after 500 steps of noise" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile after 500 steps of noising
    </div>
  </div>
  <div class="col-sm-4 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_1/1.1_noisy_750.png" title="Campanile after 750 steps of noise" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile after 750 steps of noising
    </div>
  </div>
</div>

---

### A.1.2 Classical Denoising

As a baseline comparison to diffusion-based denoising, we apply classical Gaussian blur filtering to noisy images generated at timesteps \(t \in \{250, 500, 750\}\). Gaussian denoising works by convolving the image with a Gaussian kernel, effectively averaging nearby pixels to suppress high-frequency variations that are often associated with noise. While this can slightly smooth out random fluctuations, it does not model the structure of the underlying clean image and therefore removes signal and noise indiscriminately. As the noise level increases at larger timesteps, Gaussian blur struggles even more: important edges and fine details are washed out while significant noise remains. This experiment highlights a key limitation of classical denoising methods and motivates the use of learned diffusion models, which can leverage data-driven priors to selectively remove noise while preserving semantic structure.

<div class="row justify-content-sm-center">
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_2/1.2_noisy_250.png" title="250 steps of noise" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile after 250 steps of noising
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_2/1.2_noisy_500.png" title="500 steps of noise" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile after 500 steps of noising
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_2/1.2_noisy_750.png" title="750 steps of noise" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile after 750 steps of noising
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_2/1.2_denoised_250.png" title="Denoised 250 step Campanile" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Classically denoised version (250 steps)
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/img/partA_hybrid_high_freq.png" title="Denoised 500 step Campanile" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Classically denoised version (500 steps)
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/img/partA_hybrid_high_freq.png" title="High-frequency image" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Classically denoised version (750 steps)
    </div>
  </div>
</div>

---

### A.1.3 One-Step Denoising

In this section, we leverage a pretrained diffusion model to denoise images corrupted with Gaussian noise. Specifically, we use the Stage I UNet from DeepFloyd IF, which has been trained on a massive dataset of noisy–clean image pairs to predict the noise component added at a given timestep \( t \). Given a noisy image \( x_t \), the model estimates the noise \( \epsilon_\theta(x_t, t) \), conditioned both on the timestep and a text prompt embedding (here fixed to *“a high quality photo”*). Using the forward diffusion equation, the clean image estimate is then recovered via  

\[
\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1 - \bar{\alpha}_t}\,\epsilon_\theta(x_t, t)\right),
\]

which correctly rescales the predicted noise rather than subtracting it directly. For timesteps \( t \in \{250, 500, 750\} \), we apply the forward process to add noise, pass the noisy images through the pretrained UNet to estimate noise, and reconstruct an approximation of the original image. 


<div class="row justify-content-sm-center">
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_2/1.2_noisy_250.png" title="250 steps of noise" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile after 250 steps of noising
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_2/1.2_noisy_500.png" title="500 steps of noise" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile after 500 steps of noising
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_2/1.2_noisy_750.png" title="750 steps of noise" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile after 750 steps of noising
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_3/1.3_denoised_t250.png" title="Denoised 250 step Campanile" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      One-step denoised version (250 steps)
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_3/1.3_denoised_t500.png" title="Denoised 500 step Campanile" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      One-step denoised version (500 steps)
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_3/1.3_denoised_t750.png" title="Denoised 750 step Campanile" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      One-step denoised version (750 steps)
    </div>
  </div>
</div>


---

### A.1.4 Iterative Denoising

### Iterative Denoising with Strided Timesteps

In Part 1.3, we used the diffusion forward-process equation to perform **one-step denoising**. Given a noisy image \( x_t \) at timestep \( t \), the pretrained UNet predicts the noise \( \epsilon_\theta(x_t, t) \), which we then use to estimate the clean image \( x_0 \) via

\[
\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1 - \bar{\alpha}_t}\,\epsilon_\theta(x_t, t)\right).
\]

This equation gives a **direct projection onto the clean image manifold**, but it works best only when the noise level is moderate. As \( t \) increases and the image becomes noisier, the estimate degrades.

In this section, we go further by **denoising iteratively**, which is what diffusion models are fundamentally designed to do. Rather than jumping directly from \( x_t \) to \( x_0 \), we step backward through time: from a noisier timestep \( t \) to a slightly less noisy timestep \( t' < t \). To make this efficient, we use a **strided schedule** of timesteps instead of all 1000 steps.

The update rule for one reverse step is:

\[
x_{t'} = 
\frac{\sqrt{\bar{\alpha}_{t'}}\,\beta_t}{1 - \bar{\alpha}_t}\,\hat{x}_0
\;+\;
\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t'})}{1 - \bar{\alpha}_t}\,x_t
\;+\;
v_\sigma,
\]

where:
- \( x_t \) is the current noisy image,
- \( x_{t'} \) is the next, slightly denoised image,
- \( \hat{x}_0 \) is the one-step clean image estimate from Part 1.3,
- \( \bar{\alpha}_t \) comes from the cumulative noise schedule,
- \( v_\sigma \) is additional stochastic noise (predicted by the model).

Conceptually, this equation **interpolates between the noisy image \( x_t \) and the clean estimate \( \hat{x}_0 \)**, gradually removing noise over multiple steps.  

So while Part 1.3 gave us the clean-image estimate equation, this section embeds that estimate into an **iterative reverse diffusion process**, which produces much higher-quality results—especially at high noise levels.


<div class="row justify-content-sm-center">
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_4/1.4_iter_step_30.png" title="Original Campanile" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Original Campanile.
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_4/1.4_iter_step_25.png" title="Campanile at t=90" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
    Campanile at t=90
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_4/1.4_iter_step_20.png" title="Campanile at t=240" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile at t=240
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_4/1.4_iter_step_15.png" title="Campanile at t=390" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile at t=390
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_4/1.4_iter_step_10.png" title="Campanile at t=540" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile at t=540
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_4/1.4_iter_step_5.png" title="Campanile at t=690" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile at t=690
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_4/1.4_iterative_final.png" title="Iteratively Denoised Campanile" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Iteratively Denoised Campanile
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_4/1.4_one_step.png" title="One-Step Denoised Campanile" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      One-Step Denoised Campanile
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_4/1.4_gaussian.png" title="Gaussian Blurred Campanile" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Gaussian Blurred Campanile
    </div>
  </div>
</div>

---

### A.1.5 Diffusion Model Sampling

Instead of using the diffusion model to denoise starting from some original image, in this part, I use the model starting with complete noise. Below is 5 examples of the prompt "a high quality photo", where we take i_start = 0. This is picking some random image in the image manifold based on the randomized starting noise. 


<div class="row justify-content-sm-center">
  <div class="col-sm-5 mt-5">
    {% include figure.liquid path="assets/cs180_proj5/1_5/1.5_sample_1.png" title="Sample 1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Sample 1
    </div>
  </div>
  <div class="col-sm-5 mt-5">
    {% include figure.liquid path="assets/cs180_proj5/1_5/1.5_sample_2.png" title="Sample 2" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Sample 2
    </div>
  </div>
  <div class="col-sm-5 mt-5">
    {% include figure.liquid path="assets/cs180_proj5/1_5/1.5_sample_3.png" title="Sample 3" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Sample 3
    </div>
  </div>
  <div class="col-sm-5 mt-5">
    {% include figure.liquid path="assets/cs180_proj5/1_5/1.5_sample_4.png" title="Sample 4" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Sample 4
    </div>
  </div>
  <div class="col-sm-5 mt-5">
    {% include figure.liquid path="assets/cs180_proj5/1_5/1.5_sample_1.png" title="Sample 5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Sample 5
    </div>
  </div>
</div>
</div>

---

### A.1.6 Classifier Free Guidance (CFG)

While the images produced in the previous section no longer resemble pure noise, they still lack sharp structure and often fail to depict clearly recognizable objects. This is a common issue when sampling from diffusion models using only a single conditional signal. To significantly improve image quality—at the cost of reduced diversity—we apply **Classifier-Free Guidance (CFG)**.

CFG works by combining two noise predictions at each denoising step: one **conditional** prediction \( \epsilon_c \), which uses the text prompt embedding, and one **unconditional** prediction \( \epsilon_u \), which is obtained by passing a null (empty) prompt to the model. These two predictions are then combined as

\[
\epsilon = \epsilon_u + \gamma(\epsilon_c - \epsilon_u),
\]

where \( \gamma \) is the guidance scale. When \( \gamma = 0 \), sampling is fully unconditional, and when \( \gamma = 1 \), it is equivalent to standard conditional sampling. The key improvement comes from setting \( \gamma > 1 \), which amplifies features that are strongly aligned with the text prompt, producing sharper and more coherent images.

To implement CFG, we modify the iterative denoising loop so that the UNet is evaluated **twice per timestep**—once with the conditional prompt embedding and once with an unconditional (null) embedding. The combined noise estimate is then used in the reverse diffusion update, while the variance term is taken from the conditional prediction.

Using CFG with a guidance scale of \( \gamma = 7 \) and the prompt *“a high quality photo”*, we observe a dramatic improvement in visual fidelity compared to unguided sampling. This technique forms the foundation for stronger conditioning strategies used in later sections, including more expressive prompts, visual anagrams, and hybrid images.


<div class="row justify-content-sm-center">
  <div class="col-sm-5 mt-5">
    {% include figure.liquid path="assets/cs180_proj5/1_6/1.6_cfg_sample_1.png" title="Sample 1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      CFG Sample 1
    </div>
  </div>
  <div class="col-sm-5 mt-5">
    {% include figure.liquid path="assets/cs180_proj5/1_6/1.6_cfg_sample_2.png" title="Sample 2" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      CFG Sample 2
    </div>
  </div>
  <div class="col-sm-5 mt-5">
    {% include figure.liquid path="assets/cs180_proj5/1_6/1.6_cfg_sample_3.png" title="Sample 3" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      CFG Sample 3
    </div>
  </div>
  <div class="col-sm-5 mt-5">
    {% include figure.liquid path="assets/cs180_proj5/1_6/1.6_cfg_sample_4.png" title="Sample 4" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      CFG Sample 4
    </div>
  </div>
  <div class="col-sm-5 mt-5">
    {% include figure.liquid path="assets/cs180_proj5/1_6/1.6_cfg_sample_5.png" title="Sample 5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      CFG Sample 5
    </div>
  </div>
</div>
</div>

---

### A.1.7.1 Editing Hand-Drawn and Web Images

In this section, we do the same process, but on images I drew by hand and one image from the internet.


<div class="row justify-content-sm-center">
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_avocado_i1_256.png" title="Avocado w/ i_start=1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Avocado w/ i_start=1
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_avocado_i3_256.png" title="Avocado w/ i_start=3" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Avocado w/ i_start=3
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_avocado_i5_256.png" title="Avocado w/ i_start=5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Avocado w/ i_start=5
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_avocado_i7_256.png" title="Avocado w/ i_start=7" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Avocado w/ i_start=7
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_avocado_i10_256.png" title="Avocado w/ i_start=10" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Avocado w/ i_start=10
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_avocado_i20_256.png" title="Avocado w/ i_start=20" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Avocado w/ i_start=20
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/avocado_1.7.2.png" title="Original" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Original Avocado
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing1_i1_256.png" title="Drawing 1 w/ i_start=1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 1 w/ i_start=1
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing1_i3_256.png" title="Drawing 1 w/ i_start=3" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 1 w/ i_start=3
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing1_i5_256.png" title="Drawing 1 w/ i_start=5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 1 w/ i_start=5
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing1_i7_256.png" title="Drawing 1 w/ i_start=7" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 1 w/ i_start=7
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing1_i10_256.png" title="Drawing 1 w/ i_start=10" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 1 w/ i_start=10
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing1_i20_256.png" title="Drawing 1 w/ i_start=20" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 1 w/ i_start=20
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_handdrawn_raw.png" title="Original" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Original Drawing 1
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing2_i1_256.png" title="Drawing 2 w/ i_start=1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 2 w/ i_start=1
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing2_i3_256.png" title="Drawing 2 w/ i_start=3" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 2 w/ i_start=3
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing2_i5_256.png" title="Drawing 2 w/ i_start=5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 2 w/ i_start=5
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing2_i7_256.png" title="Drawing 2 w/ i_start=7" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 2 w/ i_start=7
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing2_i10_256.png" title="Drawing 2 w/ i_start=10" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 2 w/ i_start=10
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing2_i20_256.png" title="Drawing 2 w/ i_start=20" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 2 w/ i_start=20
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_handdrawn_raw2.png" title="Original" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Original Drawing 2
    </div>
  </div>
</div>


</div>

---

### A.1.7 Image to Image Translation (SDEdit)


From this point onward, we use Classifier-Free Guidance (CFG) to improve image quality during sampling. In this section, I explore image-to-image translation using diffusion models, which enables edits tp existing images rather than generating them entirely from noise.

The core idea is simple: instead of starting the diffusion process from pure noise, we begin with a real image, add a controlled amount of noise, and then denoise it using the diffusion model. The amount of noise determines how drastic the edit will be—small noise levels preserve most of the original structure, while larger noise levels allow the model to make more significant changes. This works because the denoising process forces the noisy image back onto the manifold of natural images, requiring the model to “hallucinate” missing details in a way consistent with the learned data distribution.

This procedure follows the SDEdit algorithm. Concretely, we first apply the forward diffusion process to the original image (e.g., the Campanile) to reach a chosen timestep. Then, starting from that noisy image, we run the iterative denoising process with CFG, using a conditional text prompt (here, *“a high quality photo”*). By varying the starting timestep—using indices such as \([1, 3, 5, 7, 10, 20]\)—we obtain a sequence of edits that gradually transition from heavily modified images back toward the original.

The result is a smooth spectrum of image edits: at low noise levels, the output closely resembles the original image, while at higher noise levels, the model introduces more creative changes. This same procedure can be applied to other input images, demonstrating how diffusion models can be used as powerful, flexible tools for controlled image editing.


<div class="row justify-content-sm-center">
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_campanile_i1_256.png" title="Campanile w/ i_start=1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile w/ i_start=1
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_campanile_i3_256.png" title="Campanile w/ i_start=3" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile w/ i_start=3
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_campanile_i5_256.png" title="Campanile w/ i_start=5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile w/ i_start=5
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_campanile_i7_256.png" title="Campanile w/ i_start=7" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile w/ i_start=7
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_campanile_i10_256.png" title="Campanile w/ i_start=10" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile w/ i_start=10
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_campanile_i20_256.png" title="Campanile w/ i_start=20" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile w/ i_start=20
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_2/1.7.2campinelle_original.png" title="Original" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Original Campanile
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_landscape_i1_256.png" title="Landscape w/ i_start=1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Landscape w/ i_start=1
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_landscape_i3_256.png" title="Landscape w/ i_start=3" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Landscape w/ i_start=3
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_landscape_i5_256.png" title="Landscape w/ i_start=5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Landscape w/ i_start=5
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_landscape_i7_256.png" title="Landscape w/ i_start=7" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Landscape w/ i_start=7
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_landscape_i10_256.png" title="Landscape w/ i_start=10" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile w/ i_start=10
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_landscape_i20_256.png" title="Landscape w/ i_start=20" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Landscape w/ i_start=20
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/landscape.png" title="Original" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Original Landscape
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_temple_i1_256.png" title="Temple w/ i_start=1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Temple w/ i_start=1
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_temple_i3_256.png" title="Temple w/ i_start=3" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Temple w/ i_start=3
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_temple_i5_256.png" title="Temple w/ i_start=5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Temple w/ i_start=5
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_temple_i7_256.png" title="Temple w/ i_start=7" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Temple w/ i_start=7
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_temple_i10_256.png" title="Temple w/ i_start=10" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Temple w/ i_start=10
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1.7_temple_i20_256.png" title="Temple w/ i_start=20" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Temple w/ i_start=20
    </div>
  </div>
  <div class="col-sm-7 mt-7">
    {% include figure.liquid path="assets/cs180_proj5/1_7/temple.jpg" title="Original" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Temple Landscape
    </div>
  </div>
</div>


</div>

---

## Part B: Diffusion Models from Scratch

In Part B, I implemented and trained diffusion-style models from scratch on the MNIST dataset using **flow matching**, progressively adding time conditioning and class conditioning.

---

### B.1 Unconditional Denoising UNet

I implemented a UNet architecture trained to denoise images corrupted by Gaussian noise.

<div class="row justify-content-sm-center">
  <div class="col-sm-6 mt-3">
    {% include figure.liquid path="assets/img/partB_uncond_epoch1.png" title="Unconditional denoising (epoch 1)" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-6 mt-3">
    {% include figure.liquid path="assets/img/partB_uncond_epoch5.png" title="Unconditional denoising (epoch 5)" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

---

### B.2 Training Curve

I tracked the training loss across iterations to verify convergence.

<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3">
    {% include figure.liquid path="assets/img/partB_uncond_training_curve.png" title="Training loss curve" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

---

### B.3 Sampling from Pure Noise

I trained the model to map pure Gaussian noise to digit-like outputs, demonstrating how the model learns a data-centric average structure.

<div class="row justify-content-sm-center">
  <div class="col-sm-6 mt-3">
    {% include figure.liquid path="assets/img/partB_pure_noise_epoch1.png" title="Pure noise sampling (epoch 1)" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-6 mt-3">
    {% include figure.liquid path="assets/img/partB_pure_noise_epoch5.png" title="Pure noise sampling (epoch 5)" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

---

### B.4 Time-Conditioned Flow Matching

I extended the UNet to condition on diffusion time, allowing it to learn a continuous velocity field that transports noise to data.

<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3">
    {% include figure.liquid path="assets/img/partB_time_training_curve.png" title="Time-conditioned training curve" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

---

### B.5 Class-Conditional Flow Matching

I further extended the model to condition on digit labels using classifier-free guidance, enabling controlled generation.

<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3">
    {% include figure.liquid path="assets/img/partB_class_training_curve.png" title="Class-conditional training curve" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

---

### B.6 Class-Conditional Sampling

Finally, I sampled from the trained class-conditional model, generating four samples per digit (0–9).

<div class="row justify-content-sm-center">
  <div class="col-sm-12 mt-3">
    {% include figure.liquid path="assets/img/partB_class_samples_epoch10.png" title="Class-conditional samples (4 per class)" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

---

## Conclusion

This project demonstrates how diffusion models can be both *used* as powerful generative tools and *constructed* from first principles. By incrementally building from unconditional denoising to time- and class-conditioned flow matching, I gained a deeper understanding of how modern diffusion systems achieve controllable, high-quality generation.