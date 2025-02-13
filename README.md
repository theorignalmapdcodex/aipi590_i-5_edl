# Testing the Hypothesis of Zebra Stripe-Based Classification in GoogleNet using Saliency Map Analysis

## Table of Contents
- [Testing the Hypothesis of Zebra Stripe-Based Classification in GoogleNet using Saliency Map Analysis](#testing-the-hypothesis-of-zebra-stripe-based-classification-in-googlenet-using-saliency-map-analysis)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Goal](#project-goal)
  - [Hypothesis](#hypothesis)
  - [Implementation Steps](#implementation-steps)
    - [Data Preparation](#data-preparation)
    - [Model Selection](#model-selection)
    - [Saliency Map Generation](#saliency-map-generation)
    - [Analysis Methods](#analysis-methods)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Results](#results)
  - [References](#references)

## Overview
This project examines whether a GoogleNet model preferentially highlights the striped regions of zebras when classifying zebra images. The study utilizes saliency maps as an explainable AI technique to visualize and analyze the model's focus areas during classification.

## Project Goal
The primary objective is to demonstrate a sound experimental design and analysis process by:
- Examining the model's attention patterns on zebra images
- Using saliency maps to visualize important regions for classification
- Quantitatively analyzing the relationship between stripe regions and model attention

## Hypothesis
- **Null Hypothesis (H₀)**: The saliency maps generated for zebra images classified by the GoogleNet model do not preferentially highlight the striped regions of the zebra.
- **Alternative Hypothesis (H₁)**: The saliency maps generated for zebra images classified by the GoogleNet model preferentially highlight the striped regions of the zebra, indicating that the model focuses on these features for classification.

## Implementation Steps

### Data Preparation
1. Collection of zebra images
2. Creation of ground truth masks for stripe regions
3. Image preprocessing and resizing to 224x224 pixels

### Model Selection
- Utilizing pre-trained ResNet50 model with ImageNet weights
- Integration with TensorFlow/Keras framework

### Saliency Map Generation
The process involves:
1. Forward pass through the model
2. Gradient computation for the zebra class
3. Visualization of gradients as saliency maps
4. Overlay of saliency maps on original images

### Analysis Methods
- Bounding box generation around high-saliency regions
- IoU (Intersection over Union) calculation between saliency regions and stripe masks
- Visual comparison of saliency maps with original images

## Requirements
```
tensorflow
numpy
pandas
matplotlib
seaborn
pillow
scikit-learn
statsmodels
plotly
```

## Installation
1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Load the zebra images and corresponding masks
2. Generate saliency maps:
```python
# Generate saliency map for an image
saliency_map, top_pred_index = generate_saliency_map(model, img_array)

# Calculate IoU with stripe regions
iou = calculate_iou(bounding_box, zebra_img_mask)
```

## Results
The project provides two types of visualizations:
1. Basic saliency maps showing model attention regions
2. Overlay visualizations with bounding boxes highlighting high-saliency areas

The IoU calculations between saliency regions and stripe masks provide quantitative evidence for evaluating the hypothesis.

## References
1. [What is Saliency Map?](https://www.geeksforgeeks.org/what-is-saliency-map/)
2. [Explainable AI: Saliency Maps](https://medium.com/@bijil.subhash/explainable-ai-saliency-maps-89098e230100)
3. [MDPI - Sensors: Saliency Detection](https://www.mdpi.com/1424-8220/22/17/6516)
4. [How Could Saliency Map Help to Improve Model Performance?](https://medium.com/institute-for-applied-computational-science/how-could-saliency-map-help-to-improve-model-performance-29c2979a51ed)
   
---

📚 **Author of Notebook:** Michael Dankwah Agyeman-Prempeh [MEng. DTI '25]