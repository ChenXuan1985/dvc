"""
PyTorch 模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    """简单的全连接分类器"""
    
    def __init__(self, input_dim, hidden_layers, num_classes, dropout=0.3):
        super(SimpleClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def create_model(params):
    """根据参数创建模型"""
    input_dim = params['data']['num_features']
    hidden_layers = params['model']['hidden_layers']
    num_classes = params['data']['num_classes']
    dropout = params['model']['dropout']
    
    return SimpleClassifier(input_dim, hidden_layers, num_classes, dropout)

def save_model(model, path):
    """保存模型"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存到: {path}")

def load_model(model, path, device='cpu'):
    """加载模型"""
    model.load_state_dict(torch.load(path, map_location=device))
    return model
