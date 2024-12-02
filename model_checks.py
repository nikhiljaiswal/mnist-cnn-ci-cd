import torch.nn as nn
import torch.nn.functional as F
import json
import torch
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Input Block - Enhanced feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),    # 28x28x8
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),    # 28x28x8
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1),   # 28x28x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01)
        )
        
        # Transition Block 1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),              # 14x14x16
        )
        
        # Convolution Block 2 - Focus on distinguishing similar digits
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),  # 14x14x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),  # 14x14x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.01)
        )
        
        # Transition Block 2
        self.trans2 = nn.Sequential(
            nn.MaxPool2d(2, 2),              # 7x7x32
        )
        
        # Convolution Block 3 - Final feature refinement
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),  # 7x7x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 1),            # 7x7x16 (pointwise)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01)
        )
        
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)      # 1x1x16
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(16, 10, 1)             # 1x1x10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.trans2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = self.final(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

def load_model():
    """Load the saved model"""
    model_path = 'models/mnist_best.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please ensure the model is saved from the notebook first.")
    model = torch.load(model_path)
    model.eval()
    return model

class ModelChecker:
    def __init__(self, model):
        self.model = model
        self.checks = {
            "param_count": False,
            "batch_norm": False,
            "dropout": False,
            "gap_or_fc": False
        }
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        self.checks["param_count"] = total_params < 20000
        return total_params
    
    def check_batch_norm(self):
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.checks["batch_norm"] = True
                return True
        return False
    
    def check_dropout(self):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                self.checks["dropout"] = True
                return True
        return False
    
    def check_gap_or_fc(self):
        has_gap = False
        has_fc = False
        for module in self.model.modules():
            if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.AvgPool2d):
                has_gap = True
            if isinstance(module, nn.Linear):
                has_fc = True
        self.checks["gap_or_fc"] = has_gap or has_fc
        return has_gap or has_fc
    
    def run_all_checks(self):
        param_count = self.count_parameters()
        has_bn = self.check_batch_norm()
        has_dropout = self.check_dropout()
        has_gap_or_fc = self.check_gap_or_fc()
        
        print(json.dumps({
            "Parameter Count": param_count,
            "Under 20k Parameters": self.checks["param_count"],
            "Has BatchNorm": has_bn,
            "Has Dropout": has_dropout,
            "Has GAP or FC": has_gap_or_fc,
            "All Checks Passed": all(self.checks.values())
        }, indent=2))
        
        assert all(self.checks.values()), "Not all architecture requirements were met!"
                
if __name__ == "__main__":
    model = load_model()
    checker = ModelChecker(model)
    checker.run_all_checks() 