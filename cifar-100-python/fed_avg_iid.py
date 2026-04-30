import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from centralizedmodel import CentralizedModel
from datapreprocessing import FederatedLearningDataset

# --- 1. 本地训练逻辑 (Local Update) ---
def train_local(model, dataset_indices, full_dataset, J, batch_size, lr, device):
    """
    J 代表本地步数 (Local Steps)
    """
    model.train()
    subset = Subset(full_dataset, list(dataset_indices))
    # 为保证 J 步能取到数据，如果索引数少于 batch_size*J，loader 需要循环采样
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    steps_count = 0
    running_loss = 0.0
    
    # 模拟本地 J 步迭代
    while steps_count < J:
        for inputs, targets in loader:
            if steps_count >= J:
                break
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            steps_count += 1
            
    return model.state_dict(), running_loss / J

# --- 2. 聚合函数 ---
def aggregate_weights(local_weights_list):
    avg_weights = copy.deepcopy(local_weights_list[0])
    for key in avg_weights.keys():
        for i in range(1, len(local_weights_list)):
            avg_weights[key] += local_weights_list[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights_list))
    return avg_weights

# --- 3. 核心实验函数 ---
def run_experiment(Nc_value, J_value, is_iid=False):
    # 参数设置 [source: 1]
    K = 100           # 总客户端数
    C = 0.1           # 抽样率
    # 规则：增加 J 时，相应减少 Rounds 以保持总计算预算一致 [source: 1]
    # 设定基准：J=4, Rounds=100; J=8, Rounds=50; J=16, Rounds=25
    TOTAL_STEPS = 400 
    ROUNDS = TOTAL_STEPS // J_value
    BATCH_SIZE = 32
    LR = 0.01
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备数据 [source: 2]
    fld = FederatedLearningDataset(N=K, C=Nc_value)
    if is_iid:
        user_groups = fld.iid_partition()
    else:
        user_groups = fld.non_iid_partition()
    
    global_model = CentralizedModel(num_classes=100).to(device)
    test_loader = DataLoader(fld.test_dataset, batch_size=64, shuffle=False)
    
    best_acc = 0.0
    
    print(f"\n>> Starting: {'IID' if is_iid else f'Nc={Nc_value}'}, J={J_value}, Rounds={ROUNDS}")

    for r in range(ROUNDS):
        local_weights = []
        m = max(int(C * K), 1)
        selected_clients = np.random.choice(range(K), m, replace=False)
        
        for client_id in selected_clients:
            local_model_dict, _ = train_local(
                model=copy.deepcopy(global_model),
                dataset_indices=user_groups[client_id],
                full_dataset=fld.train_dataset,
                J=J_value,
                batch_size=BATCH_SIZE,
                lr=LR,
                device=device
            )
            local_weights.append(local_model_dict)
        
        global_weights = aggregate_weights(local_weights)
        global_model.load_state_dict(global_weights)
        
        # 评估
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = global_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        best_acc = max(best_acc, acc)
        if (r + 1) % 5 == 0:
            print(f"Round {r+1}/{ROUNDS} - Accuracy: {acc:.2f}%")

    return best_acc

# --- 4. 自动化测试脚本 ---
if __name__ == "__main__":
    results = {}
    
    # 1. 测试 IID 情况作为基准 (固定 J=4)
    # results['IID_J4'] = run_experiment(Nc_value=100, J_value=4, is_iid=True)
    
    # 2. 测试不同 Nc 和 J 的组合
    for nc in [1, 5, 10, 50]:
        for j in [4, 8, 16]:
            tag = f"Nc{nc}_J{j}"
            acc = run_experiment(Nc_value=nc, J_value=j, is_iid=False)
            results[tag] = acc
            
    # 保存结果
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nFinal Results:", results)