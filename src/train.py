"""
模型训练脚本
"""
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from model import create_model, save_model
from dataset import create_data_loaders

def load_params():
    """加载参数配置"""
    with open('params.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, params, device):
    """训练模型"""
    training_params = params['training']
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_params['learning_rate'],
        weight_decay=training_params['weight_decay']
    )
    
    early_stopping_params = training_params['early_stopping']
    patience = early_stopping_params['patience']
    min_delta = early_stopping_params['min_delta']
    
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(training_params['epochs']):
        print(f'\nEpoch {epoch + 1}/{training_params['epochs']}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            print("模型性能提升，保存最佳模型")
        else:
            epochs_no_improve += 1
            print(f"模型没有提升，早停计数: {epochs_no_improve}/{patience}")
            
            if epochs_no_improve >= patience:
                print(f'早停触发，停止训练')
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='模型训练')
    parser.add_argument('--data-dir', '-d', required=True, help='预处理数据目录')
    parser.add_argument('--model-output', '-m', required=True, help='模型输出路径')
    parser.add_argument('--checkpoint-output', '-c', required=True, help='checkpoint 输出目录')
    
    args = parser.parse_args()
    
    params = load_params()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    model = create_model(params).to(device)
    
    train_loader, val_loader = create_data_loaders(
        args.data_dir, params['training']['batch_size'])
    
    model = train_model(model, train_loader, val_loader, params, device)
    
    save_model(model, args.model_output)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'params': params
    }, f'{args.checkpoint_output}/final_checkpoint.pth')
    
    print('训练完成！')

if __name__ == '__main__':
    main()
