# DeepTrackStat

DeepTrackStat (DTS) is a novel deep learning framework for extracting motion statistics from particle videos. Developed for research in particle tracking (PT), DTS bypasses the traditional tracking process and directly estimates key motion parameters such as speed, velocity components (Vx and Vy), and turning angles from videos. This streamlined approach can reduce processing time by up to 6× compared to classical methods while maintaining performance comparable to state-of-the-art techniques.

## Overview

Particle tracking has long relied on Gaussian filtering and nearest neighbor-based algorithms to detect and link features in image sequences. However, these conventional methods struggle with high-resolution videos and sparse particle distributions where all particles share similar shapes and textures. DeepTrackStat overcomes these challenges by using deep learning models built on modern architectures (e.g., VOLO, RegNet) to extract motion statistics directly from raw image data. This approach is robust across a wide range of experimental conditions, handling significant variations in:
- **Particle size and shape**
- **Brightness and contrast**
- **Speed and trajectory complexity**
- **Particle density and signal-to-noise ratio**

DeepTrackStat has been successfully applied to diverse PT scenarios including trajectories derived from Brownian motion, Poiseuille flow, and porous media flow.

## Features

- **Speed Analysis**  
  Computes particle speed distributions by comparing deep learning model outputs with classical particle tracking methods.
- **Velocity Components**  
  Separately analyses horizontal (Vx) and vertical (Vy) velocity components.
- **Turn Angle Prediction**  
  Estimates turn angle statistics to quantify directional changes in particle movement.
- **Directionality & Dispersion**  
  Evaluates directional persistence and computes dispersion coefficients to characterize particle spread.
- **Robustness to Variability**  
  The framework is designed to work effectively with variable particle sizes, shapes, brightness, speed, density, and noise levels.
- **Visualization & Comparison**  
  Automatically generates detailed plots including probability density distributions, Earth Mover's Distance metrics, and scatter plots that compare predictions with traditional particle tracking results.

## Repository Structure

- **README.md**: This file describing the project and its context.  
- **test_simulated.py**: Scripts for analyzing simulated particle trajectories and evaluating model predictions.  
- **test_experimental.py**: Scripts for processing experimental datasets and comparing deep learning predictions with classical tracking.  
- **test.py**: Additional examples and evaluation scripts.  
- **dispersion_coeffs.py**: Code for calculating particle dispersion coefficients using linear regression over trajectory variances.  
- **dispersion_coeff_NN.ipynb**: An interactive notebook demonstrating dispersion coefficient computation.  
- **models/**: Folder containing pretrained deep learning models for speed classification, turn angle prediction, and directionality estimation.  
- **Results/**: Directory where all output figures and analysis results are saved.  
- **.gitignore**: Lists model checkpoints and other files that should be excluded from version control.

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- timm
- NumPy
- Pandas
- SciPy
- Matplotlib
- OpenCV (cv2)

### Installation

Clone the repository:

```
git clone https://github.com/yourusername/DeepTrackStat.git
```

Install the required packages:

```
pip install -r requirements.txt
```

Ensure all dependencies are installed before running the scripts.

### Usage

To analyze simulated particle videos, run:

```
python test_simulated.py
```

For experimental datasets, use:

```
python test_experimental.py
```

Additional scripts (e.g., `test.py`) are provided for further evaluation and demonstration of model performance and statistical comparisons.


## Acknowledgments

We gratefully acknowledge the contributions of the research community and the developers of PyTorch and timm, whose tools have made this work possible. Special thanks to the teams behind classical particle tracking methods whose work laid the foundation for this research.

## Contact

For further information or questions, please contact:

marc.berghouse@gmail.com

---

![DeepTrackStat Overview](https://github.com/mberghouse/DeepTrackStat/assets/55556564/24d7167e-c2c3-4932-a7f6-445e9f294800)  
![DeepTrackStat Example](https://github.com/mberghouse/DeepTrackStat/assets/55556564/6b9ae068-ca31-4ad1-b795-3204176ec993)  
![DeepTrackStat Visualization](https://github.com/mberghouse/DeepTrackStat/assets/55556564/56f561d2-cd4b-47fc-8df4-cbfbaa38603a)

 

