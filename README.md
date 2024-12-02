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
- Implements Global Average Pooling (GAP)

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

##### Train & Test the model:

Execute Jupyter Notebook file to train and test the model performance

###### Training & Test Logs

```plaintext
Epoch 1
Loss=0.5289 Batch_id=384 Accuracy=78.76%: 100%|██████████| 385/385 [00:28<00:00, 13.49it/s]
<ipython-input-5-673e9433ebf8>:53: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  misclassified_pred.append(int(pred_np[i]))
Test set: Average loss: 0.4424, Accuracy: 9646/10000 (96.46%)
Best accuracy: 96.46%
Epoch 2
Loss=0.1680 Batch_id=384 Accuracy=96.72%: 100%|██████████| 385/385 [00:25<00:00, 15.23it/s]
Test set: Average loss: 0.1454, Accuracy: 9812/10000 (98.12%)
Best accuracy: 98.12%
Epoch 3
Loss=0.0822 Batch_id=384 Accuracy=97.96%: 100%|██████████| 385/385 [00:26<00:00, 14.56it/s]
Test set: Average loss: 0.0809, Accuracy: 9872/10000 (98.72%)
Best accuracy: 98.72%
Epoch 4
Loss=0.0728 Batch_id=384 Accuracy=98.47%: 100%|██████████| 385/385 [00:26<00:00, 14.40it/s]
Test set: Average loss: 0.0600, Accuracy: 9886/10000 (98.86%)
Best accuracy: 98.86%
Epoch 5
Loss=0.0951 Batch_id=384 Accuracy=98.68%: 100%|██████████| 385/385 [00:25<00:00, 14.96it/s]
Test set: Average loss: 0.0478, Accuracy: 9892/10000 (98.92%)
Best accuracy: 98.92%
Epoch 6
Loss=0.0786 Batch_id=384 Accuracy=98.83%: 100%|██████████| 385/385 [00:25<00:00, 14.91it/s]
Test set: Average loss: 0.0406, Accuracy: 9910/10000 (99.10%)
Best accuracy: 99.10%
Epoch 7
Loss=0.0628 Batch_id=384 Accuracy=98.94%: 100%|██████████| 385/385 [00:25<00:00, 14.92it/s]
Test set: Average loss: 0.0354, Accuracy: 9911/10000 (99.11%)
Best accuracy: 99.11%
Epoch 8
Loss=0.0326 Batch_id=384 Accuracy=99.06%: 100%|██████████| 385/385 [00:26<00:00, 14.79it/s]
Test set: Average loss: 0.0301, Accuracy: 9918/10000 (99.18%)
Best accuracy: 99.18%
Epoch 9
Loss=0.0499 Batch_id=384 Accuracy=99.14%: 100%|██████████| 385/385 [00:26<00:00, 14.69it/s]
Test set: Average loss: 0.0289, Accuracy: 9919/10000 (99.19%)
Best accuracy: 99.19%
Epoch 10
Loss=0.0342 Batch_id=384 Accuracy=99.14%: 100%|██████████| 385/385 [00:26<00:00, 14.78it/s]
Test set: Average loss: 0.0262, Accuracy: 9924/10000 (99.24%)
Best accuracy: 99.24%
Epoch 11
Loss=0.0139 Batch_id=384 Accuracy=99.19%: 100%|██████████| 385/385 [00:26<00:00, 14.70it/s]
Test set: Average loss: 0.0278, Accuracy: 9919/10000 (99.19%)
Epoch 12
Loss=0.0289 Batch_id=384 Accuracy=99.29%: 100%|██████████| 385/385 [00:25<00:00, 14.92it/s]
Test set: Average loss: 0.0221, Accuracy: 9934/10000 (99.34%)
Best accuracy: 99.34%
Epoch 13
Loss=0.0037 Batch_id=384 Accuracy=99.28%: 100%|██████████| 385/385 [00:26<00:00, 14.52it/s]
Test set: Average loss: 0.0258, Accuracy: 9924/10000 (99.24%)
Epoch 14
Loss=0.0066 Batch_id=384 Accuracy=99.33%: 100%|██████████| 385/385 [00:26<00:00, 14.59it/s]
Test set: Average loss: 0.0232, Accuracy: 9937/10000 (99.37%)
Best accuracy: 99.37%
Epoch 15
Loss=0.0267 Batch_id=384 Accuracy=99.36%: 100%|██████████| 385/385 [00:26<00:00, 14.61it/s]
Test set: Average loss: 0.0256, Accuracy: 9922/10000 (99.22%)
Epoch 16
Loss=0.0214 Batch_id=384 Accuracy=99.41%: 100%|██████████| 385/385 [00:26<00:00, 14.78it/s]
Test set: Average loss: 0.0214, Accuracy: 9934/10000 (99.34%)
Epoch 17
Loss=0.0119 Batch_id=384 Accuracy=99.40%: 100%|██████████| 385/385 [00:25<00:00, 14.84it/s]
Test set: Average loss: 0.0223, Accuracy: 9931/10000 (99.31%)
Epoch 18
Loss=0.0253 Batch_id=384 Accuracy=99.47%: 100%|██████████| 385/385 [00:25<00:00, 14.86it/s]
Test set: Average loss: 0.0265, Accuracy: 9910/10000 (99.10%)
Epoch 19
Loss=0.0377 Batch_id=384 Accuracy=99.42%: 100%|██████████| 385/385 [00:26<00:00, 14.68it/s]
Test set: Average loss: 0.0221, Accuracy: 9935/10000 (99.35%)
Epoch 20
Loss=0.0070 Batch_id=384 Accuracy=99.48%: 100%|██████████| 385/385 [00:26<00:00, 14.58it/s]
Test set: Average loss: 0.0188, Accuracy: 9946/10000 (99.46%)
Best accuracy: 99.46%
```

##### Perform Model Check:

```bash
python model_checks.py 
```
 Sample Output

```plaintext
{
  "Parameter Count": 18994,
  "Under 20k Parameters": true,
  "Has BatchNorm": true,
  "Has Dropout": true,
  "Has GAP or FC": true,
  "All Checks Passed": true
}
```


## 🔄 CI/CD Pipeline

Our automated pipeline:
1. 🛠️ Sets up Python environment
2. 📦 Installs dependencies
3. 🧪 Runs all tests

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
- Run all tests

