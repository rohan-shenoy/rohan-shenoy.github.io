---
layout: page
title: Fun with Filters and Frequencies
description: Rohan Shenoy
importance: 3
category: CS180
related_publications: true
---

This project explores the fundamentals of image processing through filtering and frequency domain manipulation. We apply the techniques we develop on a series of fun images to create cool visual effects.
---

## Part 1: Fun with Filters

In this part, we will build intuitions about 2D convolutions and filtering, beginning with finite difference operators as our filters in the x and y directions.

### Part 1.1: Convolutions from Scratch!

First, let's implement convolution from scratch using numpy only. I'll compare my implementation with scipy's built-in function and analyze the differences in runtime and boundary handling.

#### What is Convolution?

**Intuitive Understanding:** Convolution is like sliding a small window (called a kernel or filter) across an image and computing a weighted sum at each position. 

For example:
- A **box filter** (all 1's normalized by the size of the kernel) computes the average of neighboring pixels, creating a blur effect by smoothing out sharp transitions. You can see this below after a box filter is applied to a grayscale version of my headdshot.
- A **finite difference operator** like `[1, 0, -1]` or `[1, 0, -1]^T` measures how much the intensity changes between adjacent pixels, highlighting the edges in the image.

The mathematical operation involves flipping the kernel and then sliding it across the image, multiplying corresponding pixel values and summing the results. This creates a new "filtered" version of the image that emphasizes certain features based on the kernel used.

#### What Does a Box Filter Do?

A box filter is one of the simplest blur filters - it's a square matrix filled with equal values (typically 1's that are then normalized). When convolved with an image, it replaces each pixel with the average of its neighbors within the box. This has a smoothing effect because sharp changes in intensity get averaged out with surrounding pixels. A 9x9 box filter takes the average of 81 neighboring pixels, creating a more pronounced blur than smaller kernels.

#### My Convolution Implementation

```python
# Four-loop implementation
def convolve_4loops(image, kernel, padding = True):
    #flip kernel to make operation commutative
    kernel = np.flip(kernel)

    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    if padding:
        pad_h = kernel_height // 2
        pad_w = kernel_width // 2
        
        padded_image = np.zeros((img_height + 2*pad_h, img_width + 2*pad_w))
        padded_image[pad_h:pad_h+img_height, pad_w:pad_w+img_width] = image
        
        result = np.zeros((img_height, img_width))
    else:
        padded_image = image
        result_height = img_height - kernel_height + 1
        result_width = img_width - kernel_width + 1
        result = np.zeros((result_height, result_width))
        pad_h, pad_w = 0, 0
    
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            conv_sum = 0
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    #convolve by manually sliding the kernel over the image
                    img_i = i + ki
                    img_j = j + kj
                    conv_sum += padded_image[img_i, img_j] * kernel[ki, kj]
            
            result[i, j] = conv_sum
    
    return result

# Two-loop implementation
def convolve_2loops(image, kernel, padding = True):
    #flip kernel
    kernel = np.flip(kernel)

    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    if padding:
        pad_h = kernel_height // 2
        pad_w = kernel_width // 2

        padded_image = np.zeros((img_height + 2*pad_h, img_width + 2*pad_w))
        padded_image[pad_h:pad_h+img_height, pad_w:pad_w+img_width] = image
        
        result = np.zeros((img_height, img_width))
    else:
        padded_image = image
        result_height = img_height - kernel_height + 1
        result_width = img_width - kernel_width + 1
        result = np.zeros((result_height, result_width))
        pad_h, pad_w = 0, 0
    
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            #convolve by getting the image matrix and multiplying element-wise by using built in numpy operations
            patch = padded_image[i:i+kernel_height, j:j+kernel_width]
            result[i, j] = np.sum(patch * kernel)
    
    return result

```

#### Comparison with scipy.signal.convolve2d

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/headshot.png" title="Original Image" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.1/box_filtered_image.jpg" title="Box Filter Result" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Original headshot image. Right: Result after applying 9x9 box filter to grayscale version of headshot.
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.1/timing_test.png" title="Timing Comparison" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Runtime comparison between my convolution implementations and scipy's built-in function.
</div>

#### Finite Difference Results

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.1/dx_filtered_image.jpg" title="Dx Filter" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.1/dy_filtered_image.jpg" title="Dy Filter" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.1/gradient_magnitude.jpg" title="Gradient Magnitude" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Result of Dx finite difference operator. Middle: Result of Dy finite difference operator. Right: Gradient magnitude image.
</div>

**Padding and Boundary Handling:**
My convolution implementation handles boundaries through zero-padding when the `padding=True` parameter is set. The algorithm calculates padding dimensions as `pad_h = kernel_height // 2` and `pad_w = kernel_width // 2`, then creates a larger array filled with zeros and places the original image in the center. This approach ensures that:

1. **Output size preservation**: With padding, the output image maintains the same dimensions as the input image
2. **Edge handling**: Pixels near image boundaries are convolved with zero-valued pixels, which can create darker edges compared to other padding strategies
3. **Valid convolution**: When `padding=False`, only valid convolutions are computed, resulting in a smaller output image with dimensions `(input_height - kernel_height + 1, input_width - kernel_width + 1)`

The zero-padding strategy is simple but can introduce boundary artifacts. 

**Runtime Comparison:**
As you can see in the runtime comparison, my two loop implementation of the convolution is an order of magnitude faster than the four loop convolution implementation. The scipy implementation of the operation is roughly 30x faster than the two loop implementation, so we use that from here on out.

---

### Part 1.2: Finite Difference Operator

Here I compute the partial derivatives of the cameraman image and create edge detection results. The intuitive approach used here is provided in the previous section.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/cameraman.png" title="Original Cameraman Image" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.2/partial_derivative_x_camerman.jpg" title="Partial Derivative X" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.2/partial_derivative_y_camerman.jpg" title="Partial Derivative Y" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption"> 
  Left: Original cameraman image. Middle: Partial derivative in x-direction. Right: Partial derivative in y-direction.
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.2/gradient_magnitude_cameraman.jpg" title="Gradient Magnitude" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.2/binary_edges_cameraman.jpg" title="Binarized Edge Image" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Gradient magnitude image, computed by taking the norm of D_x and D_y. Right: Binarized edge image.
</div>

**Threshold Selection:** To get the threshold for the binarized edge image, we play around with different threshold values until the edges come out very thin and fine, but still provide detail throughout the image. With too low of a threshold, we pick up noise in the image, and our edges become very large, and by picking too high of a threshold, we lose details from the image. We will see in the next part how to more effectively detect the edges. The threshold we use here is 35 (pixel values from [0, 255]).

---

### Part 1.3: Derivative of Gaussian (DoG) Filter

To reduce noise in edge detection, I first blur the image with a Gaussian filter before applying the finite difference operators.

#### Gaussian Smoothing First

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/cameraman.png" title="Original Cameraman" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.3/blurred_image_separable.jpg" title="Gaussian Blurred" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Original cameraman image. Right: Gaussian blurred cameraman.
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.3/dx_dog_method.jpg" title="Blurred then Dx" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.3/dy_dog_method.jpg" title="Blurred then Dy" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Dx after blurring. Right: Dy after blurring.
</div>

#### Derivative of Gaussian Filters

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.3/dog_x_filter.jpg" title="DoG X Filter" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.3/dog_y_filter.jpg" title="DoG Y Filter" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Derivative of Gaussian filter in x-direction. Right: Derivative of Gaussian filter in y-direction.
</div>

#### Single Convolution Results

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.3/gradient_dog_method.jpg" title="DoG Gradient Magnitude" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.3/gradient_smooth_method.jpg" title="Smooth Gradient Magnitude" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Gradient magnitude using DoG filters (single convolution). Right: Gradient magnitude from smooth then differentiate approach.
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/1.3/binary_edges.png" title="Binary Edge Image" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Final binary edge image using DoG filters.
</div>

**Thoughts:**
In the first approach, we begin by blurring the image and then use the finite difference operators from above. The second approach, the Derivative of Gaussian (DoG) filter approach, we convolve the 2d Gaussian kernel with the D_x/D_y kernels before convolving with the image. Because convolution is a commutative operation, we get the same results from both of these appraoches, which can be seen in the smooth gradient magnitude image vs. the DoG gradient magnitude image. Finally, we provide the binary_edges image computed from our gradient magnitude image with an intensity threshold of 35.

We see that by blurring first, we pick up much less noise throughout the image (no grass, etc).

---

## Part 2: Fun with Frequencies!

### Part 2.1: Image "Sharpening"

Using the unsharp masking technique to enhance image sharpness by emphasizing high-frequency components.

#### Theory and Implementation

The unsharp mask filter works by:
1. Creating a low-pass filtered (blurred) version of the image
2. Subtracting this from the original to get high frequencies
3. Adding scaled high frequencies back to the original image

**Formula:** `sharpened = original + α × (original - blurred)`

#### Results on Taj Mahal

We see the step by step approach to sharpening the details on the Taj Mahal. We start with a blurry original image, and create a low-pass-filter version that is even more blurry. Then we get the high-frequency image by subtracting the low-pass-filtered version from the original. Finally, we add some amount of the high_frequency image back to the original to get a sharpened version (alpha = 1.5 in this case).

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.1/taj.png" title="Original Taj" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.1/taj_blurred.jpg" title="Blurred Taj" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.1/taj_high_frequencies.jpg" title="High Frequencies" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Original Taj Mahal. Middle: Blurred version. Right: High frequency components.
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.1/taj_sharpened_alpha_1.5.jpg" title="Sharpened Taj" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Sharpened Taj Mahal using unsharp masking.
</div>

#### Additional Example

This image was taken on a hike through Peñalara, a mountain in Spain. We see how the small details in the rocks and bushes are emphasized using this technique.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.1/penalara.jpg" title="Original Peñalara" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.1/penalara_high_frequencies.jpg" title="High Frequencies Peñalara" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.1/penalara_sharpened_alpha_1.5.jpg" title="Sharpened Peñalara" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Original image. Middle: High frequency components Right: Sharpened result.
</div>

#### Evaluation: Blur then Sharpen

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.1/shenoy.jpg" title="Sharp Original" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.1/shenoy_blurred.jpg" title="Artificially Blurred" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.1/shenoy_sharpened_alpha_1.5.jpg" title="Re-sharpened" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Original sharp image. Middle: Artificially blurred. Right: Attempt to re-sharpen.
</div>

**Observations:** 
Here I try a quick experiment, in which I start with a good quality original image. I then blur it, and try recreate the "sharp" version through the process noted above. We see that the resharpened version differs from the original, because some information was lost when we apply the blur. However, in the resharpened version, you can see details on the roof of the house, as well as some added contrast in my glasses. This picture was taken in the Lofoten Islands in the North of Norway.

---

### Part 2.2: Hybrid Images

Creating hybrid images that change interpretation based on viewing distance, following the approach by Oliva, Torralba, and Schyns (SIGGRAPH 2006). Here we take a low-pass-filtered image, and then a high-pass-filtered image (taken by subtracting the low-pass image from the original) and overlay them. After overlaying them, we can see the high-pass image from up close and then the low-pass image from a distance.

#### Process Demonstration 

First, I'll demonstrate how the process works on two images of my cousins, who are twins.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/varun.jpg" title="Varun Original" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/vip.jpg" title="Viping Original" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Varun (cousin 1). Right: Vipin (cousin 2).
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.2/vipun_aligned1.jpg" title="Varun Aligned" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.2/vipun_aligned2.jpg" title="Vipin Aligned" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Varun aligned for processing. Right: Vipin aligned for processing.
</div>

#### Frequency Analysis

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.2/vipun_input1_fft.jpg" title="Varun FFT" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.2/vipun_input2_fft.jpg" title="Vipin FFT" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Fourier transform of Varun's aligned image. Right: Fourier transform of Vipin's aligned image.
</div>

#### Filtered Images and Their Frequency Content

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.2/vipun_low_freq.jpg" title="Varun Low-pass" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.2/vipun_high_freq.jpg" title="Vipin High-pass" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Varun low-pass filtered (cutoff frequency = 2.0). Right: Vipin high-pass filtered (cutoff frequency = 6.0).
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.2/vipun_lowpass_fft.jpg" title="Low-pass FFT" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.2/vipun_highpass_fft.jpg" title="High-pass FFT" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: FFT of low-pass filtered Varun. Right: FFT of high-pass filtered Vipin.
</div>

#### Final Hybrid Result

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.2/vipun_result.jpg" title="Hybrid Result" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.2/vipun_hybrid_fft.jpg" title="Hybrid FFT" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Final hybrid image of Vipin and Varun. Right: FFT of the hybrid result.
</div>

#### Understanding the Process

**How Hybrid Images Work:**
The magic of hybrid images lies in how our visual system processes different spatial frequencies. When viewing an image up close, we can resolve fine details (high frequencies), but from a distance, only the coarse structure (low frequencies) is visible.

**Frequency Domain Analysis:**
The FFT plots show the frequency content of our images in the frequency domain. The bright center represents low frequencies (smooth variations), while the outer regions represent high frequencies (sharp edges and fine details). 

**Cutoff Frequency Selection:**
- **Low-pass filter (cutoff = 2.0)**: This relatively low cutoff preserves only the basic facial structure and overall shape of Varun's face, removing fine details like individual facial features.
- **High-pass filter (cutoff = 6.0)**: This higher cutoff retains the sharp details and edges from Vipin's face while removing the broad facial structure.

The difference in cutoff frequencies (2.0 vs 6.0) creates a frequency gap that prevents interference between the two images. When combined, viewers see Vipin's detailed features up close but Varun's overall face structure from a distance, creating the hybrid effect.

**FFT Analysis:**
- The original FFT plots show the full frequency spectrum of both faces
- The low-pass FFT shows energy concentrated in the center (low frequencies only)
- The high-pass FFT shows energy in the outer regions (high frequencies only)  
- The hybrid FFT combines both, showing energy across the entire spectrum

#### Additional Hybrid Images

**Example 1: Derek + Nutmeg**

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/derekpicture.jpg" title="Derek Original" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/nutmeg.jpg" title="Nutmeg Original" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.2/derek_nutmeg_hybrid_result.jpg" title="Derek + Nutmeg Hybrid" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Derek original. Middle: Nutmeg original. Right: Derek + Nutmeg hybrid image.
</div>

**Example 2: Custom Hybrid**

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/pjs.jpg" title="Dad" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/allan.jpg" title="Dad's Coworker" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.2/pjs_allan_hybrid_result.jpg" title="Dad + Coworker Hybrid" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: My dad. Middle: My dad's coworker. Right: The hybrid result.
</div>

---

### Part 2.3 & 2.4: Multi-resolution Blending

Implementing the multi-resolution blending technique from Burt and Adelson (1983) to create seamless image composites.

#### Understanding Gaussian and Laplacian Pyramids

**Gaussian Pyramid:** A Gaussian pyramid is created by repeatedly applying a Gaussian filter and downsampling (though we maintain the same image size for visualization). Each level represents the image at progressively lower resolution, containing mainly the coarse, low-frequency information.

**Laplacian Pyramid:** The Laplacian pyramid captures the high-frequency details at each scale. Each level is computed by subtracting the next Gaussian level from the current one: `Laplacian[i] = Gaussian[i] - Gaussian[i+1]`. This gives us the band-pass filtered information at each scale.

**Multi-resolution Blending:** The key insight is that we can blend images at different frequency bands using different masks. High-frequency details need sharp transitions in the mask, while low-frequency components can use smooth mask transitions. This prevents artifacts that would occur with simple alpha blending.

#### Apple and Orange Blending Demonstration

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/apple.jpeg" title="Apple Original" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/orange.jpeg" title="Orange Original" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Left: Original apple image. Right: Original orange image.
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.3/oraple_process.png" title="Complete Pyramid Visualization" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Complete visualization showing Gaussian and Laplacian pyramids for both apple and orange images, demonstrating the multi-resolution decomposition process.
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.4/oraple_process.png" title="Blending Process Visualization" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Detailed visualization of the improved blending process for part 2.4 showing how each pyramid level is combined using the multi-resolution mask.
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.4/oraple.png" title="Final Oraple Result" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
  </div>
</div>
<div class="caption">
  The final "oraple" result - seamlessly blended apple and orange using multi-resolution techniques.
</div>

#### Additional Blending Examples

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/coffee.jpg" title="Cup of Coffee" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
  </div>
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/mars.jpg" title="Mars" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
  </div>
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.4/mars_coffee.jpg" title="Cup of Mars" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
  </div>
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/guinness.png" title="Guinness" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
  </div>
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/data/street.png" title="Nighttime Street" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
  </div>
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/proj2_results/2.4/street_guinness_blend.jpg" title="Guinness Street" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
  </div>
</div>
<div class="caption">
  Additional examples of multi-resolution blending showing input images and their seamlessly blended results.
</div>

---

## What I Learned
In this project, I learned several techniques that can be used to sharpen and blend images in different ways. Through these blending/sharpening techniques, I developed a more intuitive understanding of frequency space and how it is useful in image processing. I also had the oppurtunity to make some fun pictures!