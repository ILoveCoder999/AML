 import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from torch.utils.data import DataLoader, Subset
from centralizedmodel import CentralizedModel
from datapreprocessing import FederatedLearningDataset

# --- 1. 本地训练函数 (Local Update) ---
def train_local(model, dataset_indices, full_dataset, epochs, batch_size, lr, momentum, weight_decay, device):
    """
    执行客户端本地训练。
    项目要求 J=4 (local steps)，这里通过控制 epoch 和 batch 数量来实现。
    """
    model.train()
    subset = Subset(full_dataset, list(dataset_indices))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 根据项目要求 [cite: 59]，J=4 代表 4 个 local steps
    # 如果数据量大，只需在 loader 中迭代 4 个 batch
    steps_count = 0
    max_steps = 4 # J=4 [cite: 59]
    
    running_loss = 0.0
    for epoch in range(1): # 通常 J 很小时 1 个 epoch 足够
        for inputs, targets in loader:
            if steps_count >= max_steps:
                break
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            steps_count += 1
            
        if steps_count >= max_steps:
            break
            
    return model.state_dict(), running_loss / max_steps

# --- 2. 权重聚合函数 (Federated Averaging) ---
def aggregate_weights(local_weights_list):
    """
    简单的 FedAvg 聚合：对所有收集到的权重取平均值 [cite: 4, 151]。
    """
    avg_weights = copy.deepcopy(local_weights_list[0])
    for key in avg_weights.keys():
        for i in range(1, len(local_weights_list)):
            avg_weights[key] += local_weights_list[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights_list))
    return avg_weights

def run_fedavg_experiment():
    # configuration parameter
    N = 100           # totol clients
    C = 0.1           # 
    J = 4             # 本地步数
    ROUNDS = 50       # 通信轮数 (根据 compute budget 自行调整)
    BATCH_SIZE = 32
    LR = 0.01
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据与分片 [cite: 29, 43]
    fld = FederatedLearningDataset(N=K, C=10) # 复用你的预处理类
    user_groups = fld.iid_partition()         # 采用 IID 分片 [cite: 59]
    
    # 初始化全局模型 
    global_model = CentralizedModel(num_classes=100).to(device)
    
    # 测试集准备
    test_loader = DataLoader(fld.test_dataset, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting FedAvg: K={K}, C={C}, J={J}")

    for r in range(ROUNDS):
        local_weights = []
        m = max(int(C * K), 1) # 每轮选择的客户端数量 [cite: 62]
        selected_clients = np.random.choice(range(K), m, replace=False)
        
        round_loss = 0.0
        
        # 模拟并行：顺序执行选中客户端的本地训练 [cite: 28]
        for client_id in selected_clients:
            # 复制当前全局模型参数
            local_model_dict, loss = train_local(
                model=copy.deepcopy(global_model),
                dataset_indices=user_groups[client_id],
                full_dataset=fld.train_dataset,
                epochs=1,
                batch_size=BATCH_SIZE,
                lr=LR,
                momentum=0.9,
                weight_decay=5e-4,
                device=device
            )
            local_weights.append(local_model_dict)
            round_loss += loss
        
        # 聚合更新全局模型 [cite: 4, 151]
        global_weights = aggregate_weights(local_weights)
        global_model.load_state_dict(global_weights)
        
        # 每轮结束进行评估
        global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = global_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        print(f"Round [{r+1}/{ROUNDS}] - Avg Loss: {round_loss/m:.4f}, Test Acc: {accuracy:.2f}%")
        
        # 建议定期保存模型防止 Colab 中断 
        if (r + 1) % 10 == 0:
            torch.save(global_model.state_dict(), f'fedavg_checkpoint_r{r+1}.pth')

if __name__ == "__main__":
    run_fedavg_experiment()