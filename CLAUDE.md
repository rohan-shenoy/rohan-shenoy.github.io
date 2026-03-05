# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal academic portfolio website for Rohan Shenoy, hosted on GitHub Pages at `https://rohan-shenoy.github.io`. Plain HTML/CSS static site — no build tools or static site generators.

## Development

No build step required. Open HTML files directly in a browser or use any local server (e.g. `python3 -m http.server`). Deployed automatically via GitHub Pages from the `main` branch.

## Structure

```
index.html              # Home/About page
publications.html       # Publications list
projects.html           # Projects index (links to detail pages)
cv.html                 # Embedded PDF resume
projects/
  project1.html         # CS180 Proj 0 — Camera
  project2.html         # CS180 Proj 1 — Prokudin-Gorskii (uses MathJax)
  project3.html         # CS180 Proj 2 — Filters & Frequencies (uses MathJax)
  project4.html         # CS180 Proj 3 — Image Warping & Mosaicing
  project5.html         # CS180 Proj 4 — Neural Radiance Fields
  project6.html         # CS180 Proj 5 — Diffusion Models (uses MathJax)
assets/
  css/style.css          # Single stylesheet for the entire site
  img/headshot.jpg       # Profile photo
  pdf/rohan_resume.pdf   # CV PDF
  Proj0/                 # Project 0 images
  proj1_outputs/         # Project 1 images
  proj2_results/         # Project 2 images
  proj3/                 # Project 3 images
  proj4_deliverables/    # Project 4 images
  cs180_proj5/           # Project 5 images
```

## Conventions

- All pages share the same nav bar (copy-pasted, not templated). Project pages use `../` relative paths for nav links and assets.
- MathJax 3 CDN is included only on project pages that use LaTeX formulas (projects 2, 3, 6).
- Images use `<div class="image-row">` with optional `center` class for layout.
- CSS uses custom properties defined in `:root` in style.css for theming.
