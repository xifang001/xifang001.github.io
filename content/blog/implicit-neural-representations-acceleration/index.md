---
title: "Implicit Neural Representations"
date: 2026-09-23
lastmod: 2026-09-23
slug: "implicit-neural-representations-acceleration"
summary: "Part III: faster radiance fields through hash encodings, tensor factorization, and explicit Gaussian primitives."
description: "A study note on the computational bottleneck in NeRF and the representation changes behind Instant-NGP, TensoRF, and 3D Gaussian Splatting."
tags: [Computer Vision, Neural Fields, Implicit Neural Representations, NeRF, Instant-NGP, TensoRF, Gaussian Splatting]
categories: [Computer Vision]
series: [Implicit Neural Representations]
content_type: "Study Note"
authors: [Xi Fang]
toc: false
draft: false
math: true
part_label: "Part III"
part_summary: "Making radiance fields fast"
---

<p class="focus-label">Most radiance-field acceleration is not a new rendering equation. It is a decision about where the scene information should live.</p>

[Part II](/blog/implicit-neural-representations-nerf/) described NeRF as a continuous function queried many times along every camera ray. That representation is elegant, but its cost is immediate: a rendered image requires a large number of MLP evaluations. Part III follows the same rendering objective and asks a systems question: how can the field be represented so those queries become cheap?

Three answers mark useful points on the design space. Instant-NGP keeps a neural field but moves local detail into a multiresolution hash table [2]. TensoRF keeps volume rendering but factorizes the feature volume [3]. 3D Gaussian Splatting changes the representation more radically, replacing a neural field with optimized explicit primitives and a rasterization-style renderer [4].

## Where Vanilla NeRF Spends Its Time

For one pixel ray, NeRF evaluates <em>N</em> samples. For an image with <em>P</em> pixels, the dominant work is roughly

\[
O(PN\,C_{\mathrm{MLP}}),
\]

where <em>C</em><sub>MLP</sub> is the cost of one field query. The original coarse-to-fine scheme in Part II improves where samples go, but it still calls an MLP at every selected point. Training additionally backpropagates through those queries.

The rendering equation stays the same:

\[
\hat{\mathbf C}(\mathbf r)=\sum_i T_i\alpha_i\mathbf c_i.
\]

Acceleration therefore has several possible targets:

| Target | Question | Representative method |
| --- | --- | --- |
| Field query | Can local scene detail be looked up rather than inferred by a large MLP? | Instant-NGP |
| Field storage | Can a dense 3D feature volume be compressed without losing its structure? | TensoRF |
| Empty-space sampling | Can rays avoid regions that cannot contribute to a pixel? | Occupancy grids, used by Instant-NGP |
| Image formation | Can the scene be projected as visible primitives rather than sampled point-by-point along each ray? | 3D Gaussian Splatting |

These methods are not interchangeable implementation tricks. They make different compromises about memory, generalization, editability, and geometric interpretation.

## Instant-NGP: Put Local Detail in a Multiresolution Hash Table

An MLP that receives only a coordinate must store both global structure and local texture in its weights. Instant-NGP instead gives a small decoder a learned local feature at several spatial scales [2]. At level <em>l</em>, a coordinate <em>x</em> lies in a grid of resolution

\[
N_l=\left\lfloor N_{\min}b^l\right\rfloor,
\qquad b=\exp\!\left(\frac{\log N_{\max}-\log N_{\min}}{L-1}\right).
\]

where <em>L</em> is the number of levels. The integer grid-corner coordinate <em>i</em> does not index a full dense table. It is hashed into a bounded trainable table:

\[
h_l(\mathbf i)=\left(\bigoplus_{k=1}^{3} i_k\pi_k\right)\bmod T_l.
\]

The symbols π<sub>k</sub> are fixed large integers, ⊕ is bitwise XOR, and <em>T</em><sub>l</sub> is the number of entries in the level's table. The eight corner features are trilinearly interpolated to give <em>g</em><sub>l</sub>(<em>x</em>). Concatenating all levels produces

\[
\gamma_{\mathrm{hash}}(\mathbf x)=\bigl[\mathbf g_0(\mathbf x),\ldots,\mathbf g_{L-1}(\mathbf x)\bigr].
\]

A small MLP then predicts density and color from this encoding and the viewing direction. Hash collisions are intentional: multiple corners can share one feature vector. Coarse levels disambiguate large-scale position, interpolation enforces locality, and gradient descent separates colliding locations when the observations require it. The payoff is that high-frequency information is stored in direct, GPU-friendly feature lookups rather than repeatedly synthesized by a large network.


## Empty Space Is a Rendering Problem, Not Just a Representation Problem

Even a cheap field is slow if rays repeatedly query empty space. Instant-NGP maintains a coarse occupancy structure and samples densely only where estimated density may matter. A simple occupancy test has the form

\[
\operatorname{occupied}(v)=\mathbb{1}\!\left[\max_{\mathbf x\in v}\sigma(\mathbf x)>\tau\right].
\]

for voxel <em>v</em> and threshold τ. In practice the maximum is estimated from sampled points and updated during training. This is conservative by design: mistakenly skipping matter causes irreversible rendering errors, while visiting extra empty cells only costs time.

The lesson is broader than Instant-NGP. A high-quality radiance field usually has sparse support along a ray. Efficient methods need a way to recognize that sparsity before spending expensive field queries.

## TensoRF: Factorize the Volume Instead of Hashing It

TensoRF begins from a different observation. A dense 3D grid of learned features offers fast interpolation, but its memory grows cubically with resolution. Rather than storing every voxel independently, it decomposes the field into low-rank components [3]. A simple CP-style density factorization is

\[
\sigma(x,y,z)\approx\sum_{r=1}^{R}a_r^x(x)\,a_r^y(y)\,a_r^z(z).
\]

Each component is the product of three one-dimensional vectors. This is compact, but can be restrictive. TensoRF's vector-matrix decomposition adds plane coefficients, schematically

\[
\sigma(x,y,z)\approx\sum_{r=1}^{R}\bigl[
P_r^{xy}(x,y)v_r^z(z)+
P_r^{xz}(x,z)v_r^y(y)+
P_r^{yz}(y,z)v_r^x(x)
\bigr].
\]

The plane matrices capture correlations within coordinate pairs; the line vectors supply the remaining axis. Feature values are interpolated from these factors and passed to a small appearance decoder. The representation is still continuous at query time because the factor values are interpolated, but its learned parameters are structured arrays rather than a pure coordinate MLP.

The central trade-off is clear. Hash encodings use collision-tolerant feature tables across many scales; TensoRF uses an explicit low-rank prior. The latter can be particularly compact when scene structure is well approximated by a modest number of components, but a rank that is too small can smooth fine details.


## 3D Gaussian Splatting: Change the Primitive and the Renderer

3D Gaussian Splatting is a radiance-field method, but it is not an implicit neural representation in the strict Part I sense. The scene is an explicit set of Gaussian primitives. Primitive <em>j</em> has a mean μ<sub>j</sub>, covariance Σ<sub>j</sub>, opacity <em>o</em><sub>j</sub>, and view-dependent color parameters, commonly spherical-harmonic coefficients:

\[
\mathcal G_j=\bigl(\boldsymbol\mu_j,\Sigma_j,o_j,\mathbf q_j\bigr).
\]

For a camera projection, the 3D covariance is approximately mapped into image space using the Jacobian <em>J</em> of the perspective projection:

\[
\Sigma_j^{\prime}=J\,W\,\Sigma_j\,W^{\mathsf T}J^{\mathsf T}.
\]

where <em>W</em> transforms world coordinates into the camera frame. Nearby image pixels receive a Gaussian footprint. After depth sorting visible Gaussians, their contributions are alpha composited:

\[
\hat{\mathbf C}(\mathbf p)=\sum_j\biggl[\prod_{k=1}^{j-1}\bigl(1-\alpha_k(\mathbf p)\bigr)\biggr]\alpha_j(\mathbf p)\mathbf c_j.
\]

This looks familiar because it has the same front-to-back visibility logic as NeRF. The cost profile is different: a tile-based rasterizer projects and blends only Gaussians that cover pixels, avoiding dozens or hundreds of MLP calls per ray. Optimization alternates between fitting primitive attributes and adaptive density control—splitting, cloning, or pruning Gaussians where the current set is insufficient.


## One Task, Three Representations

| Method | Where detail lives | Main query or rendering operation | Strength | Limitation |
| --- | --- | --- | --- | --- |
| Original NeRF | MLP weights | Many MLP queries per ray | Compact, conceptually simple continuous field | Slow per-scene training and rendering |
| Instant-NGP | Multiscale hashed feature tables plus a small MLP | Hash lookup, interpolation, small decoder | Very fast optimization and rendering | Hash-table choices and GPU implementation matter; collisions are part of the representation |
| TensoRF | Low-rank plane and line factors | Interpolate factors, decode appearance | Compact structured volume with efficient queries | Rank and factorization can limit complex detail |
| 3D Gaussian Splatting | Explicit anisotropic Gaussians | Visibility-aware splatting and alpha compositing | High-quality real-time rendering | Not a coordinate MLP; large point-like primitive sets can be harder to treat as a clean surface or function |

## What Does Not Change

All four methods still depend on posed observations, multi-view consistency, and a rendering loss. They do not solve the ambiguity discussed in Part II: surfaces that are weakly observed or never observed remain uncertain. Faster rendering can make optimization practical, but it cannot create geometric evidence absent from the images.

The more subtle point is that speed changes what can be attempted. A representation that trains in minutes rather than hours can be tuned, captured, or incorporated into an interactive system much more readily. That is why the post-NeRF literature is not only about benchmarks: it is about choosing a scene representation whose computational behavior fits the application.

## References

1. B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. [*NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460392.pdf). ECCV, 2020.
2. T. Müller, A. Evans, C. Schied, and A. Keller. [*Instant Neural Graphics Primitives with a Multiresolution Hash Encoding*](https://arxiv.org/abs/2201.05989). ACM Transactions on Graphics, 2022.
3. A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su. [*TensoRF: Tensorial Radiance Fields*](https://arxiv.org/abs/2203.09517). ECCV, 2022.
4. B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis. [*3D Gaussian Splatting for Real-Time Radiance Field Rendering*](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). ACM Transactions on Graphics, 2023.
