---
layout: page
title: Fun With Diffusion Models!
description: Rohan Shenoy
importance: 6
category: CS180
related_publications: false
math: true
---

This project explores the power of diffusion models. The project is divided into two parts:  
Part A investigates the capabilities of large pretrained diffusion models for image generation and editing, while in Part B we develop and train our own diffusion model on the MNIST dataset.

---

## Part A: The Power of Diffusion Models

In Part A, I use the pretrained DeepFloyd IF, a large text-to-image diffusion model, to experiment with diffusion.

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

A core component of diffusion models is the forward diffusion process, which gradually adds noise to a clean image where the noise is Gaussian distributed. Formally, the forward process defines a conditional distribution

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\ (1-\bar{\alpha}_t)I\right),
$$

which can be equivalently written as

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0, I).
$$

Here, $$x_0$$ is the original clean image, $$x_t$$ is the noisy image at timestep $$t$$, and $$\bar{\alpha}_t$$ is a precomputed noise schedule that decreases monotonically with $$t$$. As $$t$$ increases, the signal term $$\sqrt{\bar{\alpha}_t}x_0$$ diminishes while the noise term dominates, causing the image to become progressively more corrupted. I implement this forward process to visualize how information is destroyed over time and to generate training data for the reverse denoising model, which learns to invert this corruption process by predicting clean structure from noisy inputs. We visualize this process below on an image of the Berkeley Campanile.

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

As a baseline comparison to diffusion-based denoising, we apply classical Gaussian blur filtering to noisy images generated at timesteps $$t \in \{250, 500, 750\}$$. Gaussian denoising works by convolving the image with a Gaussian kernel, effectively averaging nearby pixels to suppress high-frequency variations that are often associated with noise. While this can slightly smooth out random fluctuations, it does not model the structure of the underlying clean image and therefore removes signal and noise indiscriminately. As the noise level increases at larger timesteps, Gaussian blur struggles even more: important edges and fine details are washed out while significant noise remains. This experiment highlights a key limitation of classical denoising methods and motivates the use of learned diffusion models, which can leverage data-driven priors to selectively remove noise while preserving semantic structure.

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
    {% include figure.liquid path="assets/cs180_proj5/1_2/1.2_denoised_500.png" title="Denoised 500 step Campanile" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Classically denoised version (500 steps)
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_2/1.2_denoised_750.png" title="High-frequency image" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Classically denoised version (750 steps)
    </div>
  </div>
</div>

---

### A.1.3 One-Step Denoising

In this section, we leverage a pretrained diffusion model to denoise images corrupted with Gaussian noise. Specifically, we use the Stage I UNet from DeepFloyd IF, which has been trained on a massive dataset of noisy–clean image pairs to predict the noise component added at a given timestep $$ t $$. Given a noisy image $$ x_t $$, the model estimates the noise $$ \epsilon_\theta(x_t, t) $$, conditioned both on the timestep and a text prompt embedding (here fixed to *“a high quality photo”*). Using the forward diffusion equation, the clean image estimate is then recovered via  

$$
\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1 - \bar{\alpha}_t}\,\epsilon_\theta(x_t, t)\right),
$$

which correctly rescales the predicted noise rather than subtracting it directly. For timesteps $$ t \in \{250, 500, 750\} $$, we apply the forward process to add noise, pass the noisy images through the pretrained UNet to estimate noise, and reconstruct an approximation of the original image. 


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

In Part 1.3, we used the diffusion forward-process equation to perform one-step denoising. Given a noisy image $$ x_t $$ at timestep $$ t $$, the pretrained UNet predicts the noise $$ \epsilon_\theta(x_t, t) $$, which we then use to estimate the clean image $$ x_0 $$ via

$$
\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1 - \bar{\alpha}_t}\,\epsilon_\theta(x_t, t)\right).
$$

This equation gives a direct projection onto the clean image manifold, but it works best only when the noise level is moderate. As $$ t $$ increases and the image becomes noisier, the estimate degrades.

In this section, we go further by denoising iteratively, which is what diffusion models are fundamentally designed to do. Rather than jumping directly from $$ x_t $$ to $$ x_0 $$, we step backward through time: from a noisier timestep $$ t $$ to a slightly less noisy timestep $$ t' < t $$. To make this efficient, we use a strided schedule of timesteps instead of all 1000 steps.

The update rule for one reverse step is:

$$
x_{t'} = 
\frac{\sqrt{\bar{\alpha}_{t'}}\,\beta_t}{1 - \bar{\alpha}_t}\,\hat{x}_0
\;+\;
\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t'})}{1 - \bar{\alpha}_t}\,x_t
\;+\;
v_\sigma,
$$

where:
- $$ x_t $$ is the current noisy image,
- $$ x_{t'} $$ is the next, slightly denoised image,
- $$ \hat{x}_0 $$ is the one-step clean image estimate from Part 1.3,
- $$ \bar{\alpha}_t $$ comes from the cumulative noise schedule,
- $$ v_\sigma $$ is additional stochastic noise (predicted by the model).

Conceptually, this equation interpolates between the noisy image $$ x_t $$ and the clean estimate $$ \hat{x}_0 $$, gradually removing noise over multiple steps.  

So while Part 1.3 gave us the clean-image estimate equation, this section embeds that estimate into an iterative reverse diffusion process, which produces much higher-quality results—especially at high noise levels.


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

While the images produced in the previous section no longer resemble pure noise, they still lack sharp structure and often fail to depict clearly recognizable objects. This is a common issue when sampling from diffusion models using only a single conditional signal. To significantly improve image quality—at the cost of reduced diversity—we apply Classifier-Free Guidance (CFG).

CFG works by combining two noise predictions at each denoising step: one conditional prediction $$ \epsilon_c $$, which uses the text prompt embedding, and one unconditional prediction $$ \epsilon_u $$, which is obtained by passing a null (empty) prompt to the model. These two predictions are then combined as

$$
\epsilon = \epsilon_u + \gamma(\epsilon_c - \epsilon_u),
$$

where $$ \gamma $$ is the guidance scale. When $$ \gamma = 0 $$, sampling is fully unconditional, and when $$ \gamma = 1 $$, it is equivalent to standard conditional sampling. The key improvement comes from setting $$ \gamma > 1 $$, which amplifies features that are strongly aligned with the text prompt, producing sharper and more coherent images.

To implement CFG, we modify the iterative denoising loop so that the UNet is evaluated twice per timestep—once with the conditional prompt embedding and once with an unconditional (null) embedding. The combined noise estimate is then used in the reverse diffusion update, while the variance term is taken from the conditional prediction.

Using CFG with a guidance scale of $$ \gamma = 7 $$ and the prompt *“a high quality photo”*, we observe a dramatic improvement in visual fidelity compared to unguided sampling. This technique forms the foundation for stronger conditioning strategies used in later sections, including more expressive prompts, visual anagrams, and hybrid images.


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
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_avocado_i1_256.png" title="Avocado w/ i_start=1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Avocado w/ i_start=1
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_avocado_i3_256.png" title="Avocado w/ i_start=3" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Avocado w/ i_start=3
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_avocado_i5_256.png" title="Avocado w/ i_start=5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Avocado w/ i_start=5
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_avocado_i7_256.png" title="Avocado w/ i_start=7" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Avocado w/ i_start=7
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_avocado_i10_256.png" title="Avocado w/ i_start=10" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Avocado w/ i_start=10
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_avocado_i20_256.png" title="Avocado w/ i_start=20" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Avocado w/ i_start=20
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/avocado_1.7.2.png" title="Original" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Original Avocado
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing1_i1_256.png" title="Drawing 1 w/ i_start=1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 1 w/ i_start=1
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing1_i3_256.png" title="Drawing 1 w/ i_start=3" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 1 w/ i_start=3
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing1_i5_256.png" title="Drawing 1 w/ i_start=5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 1 w/ i_start=5
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing1_i7_256.png" title="Drawing 1 w/ i_start=7" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 1 w/ i_start=7
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing1_i10_256.png" title="Drawing 1 w/ i_start=10" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 1 w/ i_start=10
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing1_i20_256.png" title="Drawing 1 w/ i_start=20" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 1 w/ i_start=20
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_handdrawn_raw.png" title="Original" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Original Drawing 1
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing2_i1_256.png" title="Drawing 2 w/ i_start=1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 2 w/ i_start=1
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing2_i3_256.png" title="Drawing 2 w/ i_start=3" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 2 w/ i_start=3
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing2_i5_256.png" title="Drawing 2 w/ i_start=5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 2 w/ i_start=5
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing2_i7_256.png" title="Drawing 2 w/ i_start=7" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 2 w/ i_start=7
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing2_i10_256.png" title="Drawing 2 w/ i_start=10" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 2 w/ i_start=10
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_drawing2_i20_256.png" title="Drawing 2 w/ i_start=20" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Drawing 2 w/ i_start=20
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_1/1.7_handdrawn_raw2.png" title="Original" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Original Drawing 2
    </div>
  </div>
</div>


</div>

---

### A.1.7.2 Inpainting

### Inpainting with Diffusion Models (RePaint)

Inpainting is the task of filling in missing or masked regions of an image in a way that is visually consistent with the surrounding content. Using diffusion models, we can perform inpainting by slightly modifying the standard denoising process, following the ideas from the RePaint algorithm.

Given an original image $$ x_{\text{orig}} $$ and a binary mask $$ m $$, where $$ m = 1 $$ indicates regions to edit and $$ m = 0 $$ indicates regions to preserve, we run the diffusion denoising loop as usual. However, after each denoising step, we explicitly enforce that pixels outside the mask remain faithful to the original image. This is done by replacing those pixels with the appropriately noised version of the original image at timestep $$ t $$:

$$
x_t \leftarrow m \cdot x_t + (1 - m)\cdot \text{forward}(x_{\text{orig}}, t).
$$

Intuitively, this operation leaves the masked region free for the model to modify while “locking in” the unmasked region to match the original image with the correct noise level. As the denoising process progresses, the diffusion model fills in the masked region in a way that is consistent with both the surrounding pixels and the learned image manifold.

In this section, we apply inpainting to the Campanile and 2 other imags by masking out its top portion and allowing the diffusion model—guided by CFG—to generate new content in that region. We also experiment with inpainting on additional images using custom masks. Because diffusion models are being used outside of their original training distribution, multiple sampling runs may be needed to obtain a visually pleasing result.

<div class="row justify-content-sm-center">
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_2/1.7.2campinelle_original.png" title="Campanile" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_2/1.7.2campinelle_mask_applied.png" title="Infilled section" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Infilled section
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_2/1.7.2_campanile_inpaint_256.png" title="Campanile Inpainted" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Campanile Inpainted
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_2/1.7.2_basketball.png" title="Basketball" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Basketball 
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_2/1.7.2_mask_applied_basketball.png" title="Infilled section" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Infilled section
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_2/1.7.2_basketball.png" title="Basketball Inpainted" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Basketball Inpainted
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_2/1.7.2.png" title="Jump" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Jump
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_2/1.7.2_mask_applied.png" title="Infilled section" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Infilled section
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_2/1.7.2_jump_256.png" title="Jump Inpainted" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Jump Inpainted
    </div>
  </div>
</div>

---

### A.1.7.3 Text-Conditional Image-to-image Translation

In this section, I do the same procedure as in SDEdit, but I guide the project with a specified text prompt rather than using "a high quality photo". The prompts I use in this section are "a rocket ship" starting with the Campanile, "a chair growing leaves" starting with a chair in the woods, and "beer smoking a cigarette" starting with a woman smoking, respectively.

<div class="row justify-content-sm-center">
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_campinelle_rocket1_256.png" title="Rocket ship w/ i_start=1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Rocket ship w/ i_start=1
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_campinelle_rocket3_256.png" title="Rocket ship w/ i_start=3" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Rocket ship w/ i_start=3
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_campinelle_rocket5_256.png" title="Rocket ship w/ i_start=5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Rocket ship w/ i_start=5
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_campinelle_rocket7_256.png" title="Rocket ship w/ i_start=7" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Rocket ship w/ i_start=7
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_campinelle_rocket10_256.png" title="Rocket ship w/ i_start=10" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Rocket ship w/ i_start=10
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_campinelle_rocket20_256.png" title="Rocket ship w/ i_start=1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Rocket ship w/ i_start=20
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_2/1.7.2campinelle_original.png" title="Original" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Original Campanile
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_leafy_chair_i1_256.png" title="Leafy Chair w/ i_start=1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      A Chair Growing Leaves w/ i_start=1
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_leafy_chair_i3_256.png" title="Leafy Chair w/ i_start=3" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      A Chair Growing Leaves w/ i_start=3
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_leafy_chair_i5_256.png" title="Leafy Chair w/ i_start=5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      A Chair Growing Leaves w/ i_start=5
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_leafy_chair_i7_256.png" title="Leafy Chair w/ i_start=7" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      A Chair Growing Leaves w/ i_start=7
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_leafy_chair_i10_256.png" title="Leafy Chair w/ i_start=10" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      A Chair Growing Leaves w/ i_start=10
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_leafy_chair_i20_256.png" title="Leafy Chair w/ i_start=20" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      A Chair Growing Leaves w/ i_start=20
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_leafy_chair.png" title="Original" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Original Starting Image of a Chair in Leaves
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_beer_cig_i1_256.png" title="Beer Cig" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Beer Smoking a Cigarette w/ i_start=1
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_beer_cig_i3_256.png" title="Beer Cig" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Beer Smoking a Cigarette w/ i_start=3
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_beer_cig_i5_256.png" title="Beer Cig" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Beer Smoking a Cigarette w/ i_start=5
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_beer_cig_i7_256.png" title="Beer Cig" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Beer Smoking a Cigarette w/ i_start=7
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_beer_cig_i10_256.png" title="Beer Cig" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Beer Smoking a Cigarette w/ i_start=10
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3_beer_cig_i20_256.png" title="Beer Cig" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Beer Smoking a Cigarette w/ i_start=20
    </div>
  </div>
  <div class="col-sm-2 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_7/1_7_3/1.7.3smoking.png" title="Beer Cig" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Original Starting Image of a Woman Smoking
    </div>
  </div>
</div>

---

### A.1.8 Visual Anagrams with Diffusion Models

In this section, we create visual anagrams—optical illusions where a single image conveys two different meanings depending on how it is viewed. Using diffusion models, we can construct an image that appears as one concept when upright, but reveals a completely different interpretation when flipped upside down.

The key idea is to guide the denoising process using two different text prompts simultaneously. At each diffusion step, we compute two noise estimates:
- $$ \epsilon_1 $$: obtained by denoising the image normally using prompt $$ p_1 $$,
- $$ \epsilon_2 $$: obtained by flipping the image upside down, denoising it with prompt $$ p_2 $$, and then flipping the predicted noise back.

Formally, the procedure at timestep $$ t $$ is:
$$
\epsilon_1 = \text{CFG}(\text{UNet}(x_t, t, p_1)),
$$
$$
\epsilon_2 = \text{flip}\big(\text{CFG}(\text{UNet}(\text{flip}(x_t), t, p_2))\big),
$$
$$
\epsilon = \frac{\epsilon_1 + \epsilon_2}{2}.
$$

The averaged noise estimate $$ \epsilon $$ is then used in the reverse diffusion update. This forces the generated image to satisfy both prompt constraints simultaneously—one in the original orientation and one in the flipped orientation.

As a result, the final image lies at an intersection of two semantic manifolds: when viewed normally, it aligns with prompt $$ p_1 $$, and when flipped upside down, it aligns with prompt $$ p_2 $$.

<div class="row justify-content-sm-center">
  <div class="col-sm- mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_8/1.8_oil_painting_flipped_256.png" title="An Oil Painting of an Old Man" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      An Oil Painting of an Old Man
    </div>
  </div>
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_8/1.8_oil_painting_256.png" title="An Oil Painting of People around a Campfire" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      An Oil Painting of People around a Campfire
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-4 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_8/1.8_skull_256.png" title="A Skull" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Skull
    </div>
  </div>
  <div class="col-sm-4 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_8/1.8_skull_flipped_256.png" title="An Oil Painting of a Village in the Mountains" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      An Oil Painting of a Village in the Mountains
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-4 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_8/1.8_goldfish_coast.png" title="The Amalfi Coast" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      The Amalfi Coast
    </div>
  </div>
  <div class="col-sm-4 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_8/1.8_goldfish_coast_flipped_256.png" title="A Goldfish in a Wine Glass" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      A Goldfish in a Wine Glass
    </div>
  </div>
</div>

---

### A.1.9 Hybrid Images


In this part of the project, I generate hybrid images using a diffusion-based technique known as factorized diffusion. The goal is to create a single image that encodes two different semantic concepts at different spatial frequency bands—one that dominates at low frequencies and another that appears primarily at high frequencies. As a result, the image can be interpreted differently depending on viewing distance or resolution.

The method works by leveraging the diffusion model’s noise prediction mechanism. At each denoising step, I compute two classifier-free guided noise estimates using two different text prompts. One noise estimate is filtered with a low-pass Gaussian blur to retain only coarse, global structure, while the other is filtered with a high-pass operation to retain fine details. These two filtered noise signals are then added together to form a composite noise estimate, which is used in the reverse diffusion update.

This approach is closely related to how visual anagrams work earlier in the project: instead of enforcing consistency under spatial transformations like flipping, hybrid images enforce consistency across frequency decompositions. By controlling which prompt influences which frequency band, the diffusion process produces images that smoothly fuse two semantic interpretations into a single coherent output. The result demonstrates how diffusion models can be guided not just semantically, but also structurally, by manipulating the noise they remove at each step.

<div class="row justify-content-sm-center">
  <div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_9/1.9_pencil_painting_256.png" title="A Pencil Tip + People Camping" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      A Pencil Tip + People Camping
    </div>
  </div>
</div>
<div class="col-sm-3 mt-3">
    {% include figure.liquid path="assets/cs180_proj5/1_9/1.9_rocket_waterfall_256.png" title="A Rocket Ship + Waterfall" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      A Rocket Ship + Waterfall
    </div>
  </div>
</div>

---

### B.1 Single-Step Denoising UNet

In this part, we will build a training pipeline and train a UNet which will be a diffusion model that creates the digits as in the MNIST dataset. We will try a few different methods and compare which works best.

---

### B.1.2 Training a UNet Denoiser

In this section, I train a UNet to act as a denoiser by learning a direct mapping from noisy images to clean images. Given a clean MNIST digit $$ x $$, I generate a noisy version $$ z $$ by adding Gaussian noise according to
$$
z = x + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I),
$$
where $$ \sigma $$ controls the noise level. The UNet $$ D_\theta $$ is trained to recover the original image by minimizing a mean squared error objective,
$$
\mathcal{L} = \mathbb{E}_{x,z}\big[\|D_\theta(z) - x\|^2\big].
$$
By visualizing the noising process for increasing values of $$ \sigma $$, we observe progressively stronger corruption of the image, which motivates learning a data-driven denoiser rather than relying on fixed filters.

<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/2_2/1.2_noisy_images.png" title="Visualizations of noise added across different sigma values." class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Visualizations of noise added across different sigma values.
    </div>
  </div>
</div>

---

### B.1.2.1 Training Pipeline

In this section, I train a UNet-based denoising model to map a noisy image $$ z $$ back to its clean version $$ x $$. The training objective is to minimize the mean squared error between the network output and the clean image,
$$
\mathcal{L} = \mathbb{E}_{x,\epsilon}\big[\lVert D_\theta(x + \sigma \epsilon) - x \rVert^2\big],
$$
where noise is added according to $$ z = x + \sigma \epsilon $$ with $$ \epsilon \sim \mathcal{N}(0, I) $$. During training, noise is applied *on the fly* to each batch so that the model sees a different noisy version of the same image every epoch, improving generalization.

**Hyperparameters and setup:**
- **Dataset:** MNIST (training split only), shuffled each epoch  
- **Noise level: $$ \sigma = 0.5 $$, fixed throughout training  
- **Model:** UNet with hidden dimension $$ D = 128 $$  
- **Batch size:** 256  
- **Optimizer:** Adam  
- **Learning rate:** $$ 1 \times 10^{-4} $$  
- **Epochs:** 5  

After training, I visualize denoising results on the test set after the 1st and 5th epochs, as well as the training loss curve over iterations. 

<div class="row justify-content-sm-center">
  <div class="col-sm-4 mt-2">
    {% include figure.liquid path="assets/cs180_proj5/2_2/2_2epoch1.png" title="Visualizations of predictions after 1 epoch" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Visualizations of predictions after 1 epoch
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-4 mt-2">
    {% include figure.liquid path="assets/cs180_proj5/2_2/2_2epoch5.png" title="Visualizations of predictions after 5 epochs" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Visualizations of predictions after 5 epochs
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/2_2/2_2training_curve.png" title="Training Curve" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Training curve
    </div>
  </div>
</div>

---

### B.1.2.2 Out-of-Distribution Testing

In this section, I test out the model that was trained in the previous section across differnet sigma values. This will test the robustness of the model to generalize to different noise levels, despite the model only being trained for sigma = 0.5.

<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/2_2/2_2ood_denoising.png" title="Visualizations of model predictions across different sigma values" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Visualizations of model predictions across different sigma values.
    </div>
  </div>
</div>

---

### B.1.2.3 Denousing Pure Noise

Now I retry the training process, howoever, this time I input pure noise into the model to denoise it. Since I use MSE loss, the model will learn to output the image the is closest to all images in the data set. This will make the model learn to output something that looks like an eight, with a weaker bottom left side. This is because if we trace a heat map of all the numbers in the dataset, they will overlap in most parts, but the least in the bottom left part of the "eight".

<div class="row justify-content-sm-center">
  <div class="col-sm-4 mt-2">
    {% include figure.liquid path="assets/cs180_proj5/2_2/2_3_Pure_Noise_Epoch1.png" title="Visualizations of predictions after 1 epoch" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Visualizations of predictions after 1 epoch
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-4 mt-2">
    {% include figure.liquid path="assets/cs180_proj5/2_2/2_3_Pure_Noise_Epoch5.png" title="Visualizations of predictions after 5 epochs" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Visualizations of predictions after 5 epochs
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/2_2/2_2training_curve.png" title="Training Curve" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Training curve
    </div>
  </div>
</div>

---

### B.2 Conditioning the UNet

In the previous section, we saw that one-step denoising alone is insufficient for high-quality generation, motivating a truly iterative generative process. In Part 2, we switch to **flow matching**, a formulation that directly learns how to transport noise into data by modeling a continuous vector field. Instead of viewing generation as discrete denoising steps, we interpolate between pure noise $$ x_0 \sim \mathcal{N}(0, I) $$ and a clean data sample $$ x_1 $$ via $$ x_t = (1 - t)x_0 + t x_1 $$. This defines a trajectory through image space, and the key idea is to train a UNet to predict the *flow*—the instantaneous velocity $$ u(x_t, t) = \frac{d}{dt} x_t = x_1 - x_0 $$—that moves a point along this path from noise toward data. By minimizing a simple regression loss between the predicted and true flow, the model learns a continuous dynamical system that can be integrated at sampling time to generate images from noise, naturally enabling iterative generation without relying on explicit diffusion noise schedules. 

---

### B.2.2 Training the Time Conditioned UNet

To enable time conditioning in the UNet, we inject the scalar timestep $$ t \in [0,1] $$ directly into the network so that its predictions depend explicitly on where the input lies along the noise–data trajectory. Intuitively, the model needs to behave very differently when the image is close to pure noise versus when it is nearly clean, and time conditioning provides this context. We implement this by passing the normalized timestep through small fully connected blocks (FCBlocks) that map the scalar $$ t $$ to channel-wise modulation vectors. These vectors are then used to *scale* intermediate feature maps inside the UNet—specifically after the bottleneck “unflatten” stage and in the decoder—so the network’s activations are multiplicatively adjusted based on time. Rather than predicting the clean image directly, the time-conditioned UNet learns a time-dependent flow field that tells us how to move from a noisy sample toward the data manifold, making time conditioning essential for accurate iterative generation.

To train the time-conditioned UNet, we follow a simple flow-matching procedure. For each training step, we sample a clean MNIST image $$ x_1 $$ and a random timestep $$ t \sim \text{Uniform}(0,1) $$. We then generate an intermediate noisy image $$ x_t $$ by interpolating between pure noise and the clean image, and compute the target **flow** $$ u(x_t, t) = x_1 - x_0 $$, which represents the direction the model should move to reach the data manifold. The time-conditioned UNet takes $$ (x_t, t) $$ as input and is trained to predict this flow using a mean squared error loss. Crucially, noise is added *on the fly* when batches are fetched so the model sees different noisy versions of the same image across epochs. We train on the MNIST training set using a UNet with hidden dimension $$ D=64 $$, the Adam optimizer with an initial learning rate of $$10^{-2}$$, and an exponential learning rate decay applied once per epoch. The result is a model that learns a time-dependent vector field guiding samples from noise toward clean digits, which we monitor using a training loss curve over the full training process.



<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/3_2/2.2_training_curve.png" title="Training Curve" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Training curve
    </div>
  </div>
</div>

---

### B.2.3 Sampling from the UNet

After training, we can use the time-conditioned UNet as a generative model by running the *reverse diffusion process*. Sampling begins from pure Gaussian noise $$ x_T \sim \mathcal{N}(0, I) $$ and iteratively denoises the image over timesteps $$ t = T, \dots, 1 $$. At each step, the UNet takes the current noisy image $$ x_t $$ and the normalized timestep $$ t $$ as input and predicts the **flow** (i.e., the direction that moves the image toward a cleaner version). Using this predicted flow, we update the image to obtain $$ x_{t-1} $$, gradually removing noise. Repeating this process produces increasingly structured images. As expected, samples improve as training progresses: early-epoch models (e.g., epoch 1) produce noisy or incomplete digits, while later checkpoints (epochs 5 and 10) generate much clearer results. However, notice that many of the digits are not actual digits, and just form digit like shapes.

<div class="row justify-content-sm-center">
  <div class="col-sm-10 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/3_3/2.3_samples_epoch_1.png" title="Epoch 1" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Epoch 1
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-10 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/3_3/2.3_samples_epoch_5.png" title="Epoch 5" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Epoch 5
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-10 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/3_3/2.3_samples_epoch_10.png" title="Epoch 10" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Epoch 10
    </div>
  </div>
</div>

---

### B.2.5 Class Conditioning

To gain finer control over the generation process, we extend the time-conditioned UNet to also condition on the digit class $$ c \in \{0,\dots,9\} $$. Instead of representing $$ c $$ as a scalar, we encode it as a one-hot vector and inject it into the network using additional fully connected blocks (FCBlocks), analogous to time conditioning. Concretely, both the time variable $$ t $$ and class vector $$ c $$ are embedded via learned linear projections and used to *modulate* intermediate feature maps through affine transformations of the form $$ \text{feature} \leftarrow c_i \cdot \text{feature} + t_i $$. This allows the UNet to adapt its denoising behavior based on both noise level and desired class. To preserve the ability to sample unconditionally and enable classifier-free guidance, we randomly drop the class conditioning vector with probability $$ p_{\text{uncond}} = 0.1 $$ during training by setting it to zero. 

The training process is the same as the time-only, but we occasionally do unconditional generation. In the first training loop, we use a learning rate scheduler: torch.optim.lr_scheduler.ExponentialLR. In the second loop, we attempt to avoid the learning rate scheduler, and to make up for the increased training overhead, we drop the learning rate to 0.003 from 0.01.

<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/3_5/2.5_training_curve.png" title="Training curve with learning rate scheduler" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Training curve with learning rate scheduler
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/3_5/2.5_no_lr_training_curve.png" title="Training curve without learning rate scheduler" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Training curve without learning rate scheduler
    </div>
  </div>
</div>

---

### B.2.6 Class Conditioning Sampling

To sample from the class-conditioned UNet, we start from pure Gaussian noise and iteratively denoise using the reverse diffusion process, conditioning on both the timestep $$ t $$ and a target digit class $$ c $$. At each timestep, we perform **classifier-free guidance (CFG)** by evaluating the UNet twice: once with the class condition $$ c $$ and once with the class condition dropped (i.e., unconditional). These two predicted flows are combined as

$$
v = v_{\text{uncond}} + \gamma \bigl(v_{\text{cond}} - v_{\text{uncond}}\bigr),
$$

where $$ \gamma $$ is the guidance scale. This amplifies class-specific features while still leveraging the model’s learned unconditional prior. The guided flow is then used to update the noisy image according to the reverse diffusion update rule, stepping from timestep $$ t $$ to $$ t-1 $$. Repeating this process over all timesteps gradually transforms noise into a digit that strongly matches the specified class. By adjusting $$ \gamma $$, we can trade off diversity for fidelity: larger values produce clearer, more class-consistent digits.

<div class="row justify-content-sm-center">
  <div class="col-sm-10 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/3_6/2.6_samples_epoch_1.png" title="Epoch 1 Samples" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Epoch 1 Samples with learning rate scheduler
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-10 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/3_6/2.6_samples_epoch_5.png" title="Epoch 5 Samples" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Epoch 5 Samples with learning rate scheduler
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-10 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/3_6/2.6_samples_epoch_10.png" title="Epoch 10 Samples" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Epoch 10 Samples with learning rate scheduler
    </div>
  </div>
</div>

<div class="row justify-content-sm-center">
  <div class="col-sm-10 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/3_6/2.6_samples_no_lr_epoch_1.png" title="Epoch 1 Samples" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Epoch 1 Samples without learning rate scheduler
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-10 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/3_6/2.6_samples_no_lr_epoch_5.png" title="Epoch 5 Samples" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Epoch 5 Samples without learning rate scheduler
    </div>
  </div>
</div>
<div class="row justify-content-sm-center">
  <div class="col-sm-10 mt-1">
    {% include figure.liquid path="assets/cs180_proj5/3_6/2.6_samples_no_lr_epoch_10.png" title="Epoch 10 Samples" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
      Epoch 10 Samples without learning rate scheduler
    </div>
  </div>
</div>

As we can see, training without the learning rate scheduler did not significantly impact performance (this can also be seen in the training curve).

---