import torch
import numpy as np
import json
import copy
from torch.utils.data import DataLoader
from centralizedmodel import CentralizedModel, evaluate
from datapreprocessing import FederatedLearningDataset
from fed_iid import train_local, aggregate_weights
from centralizedmodel import LabelSmoothingCrossEntropy

def run_fed_non_iid_experiment(Nc_value, J_value):

    K = 100
    C = 0.1
    J = 4

    # 逻辑调整：为了公平比较，固定“总通信预算”或“总计算步数”
    # 项目书建议：When increasing J, scale accordingly the number of training rounds
    # 我们设定基础：J=4 时跑 100 轮。那么 J=16 时（计算量4倍），轮数应为 25 轮。
    # 或者为了观察收敛，设定一个基准总步数（例如 400 步）
    ROUNDS = 200

    BATCH_SIZE = 128
    LR = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1 datapreprocessing
    fld = FederatedLearningDataset(N=K, C=Nc_value)
    user_groups =  fld.non_iid_partition()

    # 2 model initialization
    global_model = CentralizedModel(num_classes=100).to(device)
    test_loader = DataLoader(fld.test_dataset, batch_size=64, shuffle=False)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': []
    }
    best_acc = 0.0
    print(f"\n>> Running: { f'Nc={Nc_value}'}, J={J_value}")



    # 3. federated learning train loop
    for r in range(ROUNDS):
        unfreeze_round = ROUNDS // 4
        if r == unfreeze_round:
            print(f"\n>>> Round {r + 1}: Unfreezing the backbone for fine-tuning! <<<")
            for param in global_model.backbone.parameters():
                param.requires_grad = True
        # 动态学习率衰减逻辑
        current_lr = LR
        if r >= unfreeze_round:
            current_lr = LR * 0.1
        if r >= ROUNDS * 0.6:
            current_lr = LR * 0.01
        if r >= ROUNDS * 0.8:
            current_lr = LR * 0.001
        local_weights = []
        m = max(int(C * K), 1)
        selected_clients = np.random.choice(range(K), m, replace=False)

        for client_id in selected_clients:
            # directly call fed_iid train function
            w, _= train_local(
                model=copy.deepcopy(global_model),
                dataset_indices=user_groups[client_id],
                full_dataset=fld.train_dataset,
                J=J_value,
                batch_size=BATCH_SIZE,
                lr=LR,
                momentum=0.9,
                weight_decay=5e-4,
                device=device
            )
            local_weights.append(w)

        # directly call fed_iid aggregation function
        global_model.load_state_dict(aggregate_weights(local_weights))

        # evaluate (call centralizedmodel evaluate function)
        test_loss, acc = evaluate(global_model, test_loader, criterion, device)
        best_acc = max(best_acc, acc)

        # 记录数据 (强烈建议套上 float()，防止 PyTorch Tensor 导致后续 JSON 序列化报错)
        avg_train_loss = test_loss/ m
        history['train_loss'].append(float(avg_train_loss))
        history['test_loss'].append(float(test_loss))  # 新增这行
        history['test_acc'].append(float(acc))

        # 修改打印逻辑：顺便把 test_loss 也打印到控制台
        if (r + 1) % 5 == 0:
            print(f"Round {r + 1}/{ROUNDS} | Train Loss: {avg_train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {acc:.2f}% | Best Acc: {best_acc:.2f}%")

    return history


if __name__ == "__main__":
    results= run_fed_non_iid_experiment(1,4)
    with open('non_iid_history_Nc1_J4.json', 'w') as f:
        json.dump(results, f, indent=4)