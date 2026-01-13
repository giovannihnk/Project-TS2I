# Image Compression using Discrete Cosine Transform (DCT)

This project focuses on the implementation of a digital image compression and decompression algorithm using the **Discrete Cosine Transform (DCT)**, a specific application of Fourier analysis. Developed as part of a Signal and Image Processing course at **Polytech Nice Sophia - Université Côte d'Azur**.

## Overview

The program utilizes Python to compress and decompress digital images by transforming spatial data into frequency components. By decomposing images into $8 \times 8$ pixel blocks and applying a frequency base change, the algorithm effectively reduces file size by eliminating high-frequency information that is less perceptible to the human eye.



## Key Features

* **DCT-II Implementation:** Uses the Discrete Cosine Transform to switch from the spatial domain to the frequency domain.
* **RGB Support:** Processes standard Red, Green, and Blue color channels by decomposing the image into three planes.
* **Low-Pass Filtering:** Includes a configurable low-pass filter to remove noise and high-frequency data based on a cutoff frequency $\omega_c$.
* **Quantization:** Implements a quantization matrix $Q$ to optimize compression rates. The intensity of compression can be adjusted by a factor $k$.
* **Performance Metrics:** Automatically calculates the compression ratio and the relative error using the Frobenius norm.

## Algorithm Workflow

1.  **Initialization:** The image is truncated to multiples of 8, and intensities are centered from $[0, 255]$ to $[-128, 127]$.
2.  **Compression:**
    * The image is divided into $8 \times 8$ blocks.
    * Change of base: $D = P M P^T$ (where $P$ is the DCT matrix).
    * Quantization: $D$ is divided by $Q$ and rounded to the nearest integer to eliminate high frequencies.
3.  **Decompression:**
    * Inverse quantization: Multiply the coefficients back by $Q$.
    * Inverse DCT: Apply $M = P^T D P$ to return to the spatial domain.
    * Data recentering ($+128$) and normalization.

## Results & Performance

The project demonstrates that compression is highly effective on simple or uniform images compared to highly textured ones.

| Image Type | Compression Rate | Relative Error |
| :--- | :--- | :--- |
| Low Texture (e.g., Waves) | ~96.5% | ~1.8% |
| High Texture (e.g., Clovers) | ~66.2% | ~13.8% |

### Execution Time
* **JPEG Processing:** Generally faster (~0.04s compression) as the format is already optimized for lossy compression.
* **PNG Processing:** Takes longer (~0.11s compression) due to the higher information density of lossless formats.
* **Decompression:** Consistently faster than compression across all tests as it requires fewer analytical calculations.

## Requirements

* Python 3.x
* NumPy (for matrix operations)
* Matplotlib (for results visualization)

## Authors
* **Ben Khalifa Emna**
* **Costantin Perline**
* **Honakoko Giovanni**
* **Zouarhi Yassmin**

---
**Course:** Signal and Image Processing (January 2025)
