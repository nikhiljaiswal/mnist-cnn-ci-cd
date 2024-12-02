import torch.nn as nn
import json
import torch
import os

def load_model():
    """Load the saved model instead of importing from notebook"""
    model_path = 'models/mnist_best.pth'  # or whatever path you save your model to
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