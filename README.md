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
- **Robustness to Variability**  
  The framework is designed to work effectively with variable particle sizes, shapes, brightness, speed, density, and noise levels.
- **Visualization & Comparison**  
  Automatically generates detailed plots including probability density distributions, Earth Mover's Distance metrics, and scatter plots that compare predictions with traditional particle tracking results.

## Repository Structure

- **README.md**: Project overview and documentation
- **testing/**: Main directory containing test scripts and evaluation code
  - **uncalibrated_test.py**: Evaluation script for uncalibrated model performance
  - **calibrated_test.py**: Evaluation script for calibrated model performance
  - **old_test_scripts/**: Archive of previous testing implementations
    - **test.py**: Original testing framework
    - **test_simulated.py**: Evaluation on simulated particle trajectories
    - **test_experimental.py**: Evaluation on experimental datasets
- **training/**: Main directory containing training notebooks
- **models/**: Folder that contains all 93 pretrained deep learning models. You must download the models from https://zenodo.org/records/14879357.
- **Results/**: Generated outputs and visualizations
- **.gitignore**: Version control exclusions

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

Ensure all prerequisites listed above are installed before running the scripts.

### Usage

First, go to https://zenodo.org/records/14879357 and download all the model weights for DTS. Place these model weights in a folder called models that will have the path "DeepTrackStat/models/"

To analyze simulated particle videos with the calibrated model, run:

```
python testing/calibrated_test.py
```

For the uncalibrated model, use:

```
python testing/uncalibrated_test.py
```

For training, please use the notebooks in the "training" folder as a guide. The basic idea is that you need to calculate the statistics from the ground truth trajectories and train the model to predict these statistics given the respective imagery.


## Acknowledgments

This research is based upon work supported by the U.S. Department of Energy (DOE) under award number DE-SC0019437.

## Contact

For further information or questions, please contact:

marc.berghouse@gmail.com

---

![DeepTrackStat Overview](https://github.com/mberghouse/DeepTrackStat/assets/55556564/24d7167e-c2c3-4932-a7f6-445e9f294800)  
![DeepTrackStat Example](https://github.com/mberghouse/DeepTrackStat/assets/55556564/6b9ae068-ca31-4ad1-b795-3204176ec993)  
![DeepTrackStat Visualization](https://github.com/mberghouse/DeepTrackStat/assets/55556564/56f561d2-cd4b-47fc-8df4-cbfbaa38603a)

 

