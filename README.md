# 🤖 MNIST CNN Model CI/CD Pipeline

![Build Status](https://github.com/nikhiljaiswal/ml-ci_cd-pipeline/actions/workflows/ml_pipeline.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.8-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A production-ready CNN model for MNIST digit classification with automated CI/CD pipeline integration.

## 🏗 Model Architecture ️

The model follows a carefully designed architecture with the following key components:

### 📊 Network Design 
- Input Block: Enhanced feature extraction (1→8→8→16 channels)
- Middle Block: Feature refinement (16→16→32 channels)
- Final Block: Feature consolidation (32→16→10 channels)
- Global Average Pooling (GAP) for final feature maps


 
## 🎯 Model Requirements
- Parameters: < 20,000
- Accuracy: > 99.4% on test set
- Training: < 20 epochs
- Uses BatchNormalization
- Implements Dropout
- Global Average Pooling (GAP)

## 🚀 Quick Start

### 1. Local Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```
### 2. Training & Testing

##### Train the model:

Execute Jupyter Notebook file to train and test the model performance

##### Run tests (two options):

a. Using pytest (detailed test results):

```bash
pytest -v test_model.py
```
 Sample Output

```plaintext
================================= test session starts =================================
platform linux -- Python 3.10.9, pytest-8.3.3, pluggy-1.5.0
collected 3 items
test_model.py::test_model_parameters PASSED [ 33%]
test_model.py::test_input_output_shape PASSED [ 66%]
test_model.py::test_model_accuracy PASSED [100%]
```


## 🔄 CI/CD Pipeline

Our automated pipeline:
1. 🛠️ Sets up Python environment
2. 📦 Installs dependencies
3. 🎯 Trains the model
4. ✅ Validates architecture and performance
5. 🧪 Runs all tests
6. 💾 Creates timestamped model artifact

This repository includes automated checks for:
- Parameter count limit
- Use of BatchNormalization
- Implementation of Dropout
- Presence of GAP/FC layer

## 🚀 Deployment

1. Create a new GitHub repository

2. Initialize git in your local project:

```bash
git init
git add .
git commit -m "Initial commit"
``` 

3. Add your GitHub repository as remote and push:

```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

The GitHub Actions workflow will automatically run when you push to the repository. It will:
- Set up a Python environment
- Install dependencies
- Train the model
- Run all tests
- Save the model as an artifact

