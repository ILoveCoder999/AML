import os.path

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
    J = 8

    #固定“总通信预算”或“总计算步数”
    #When increasing J, scale accordingly the number of training rounds
    #J=4,round=200,J=8,round=100,J=16,round=50

    ROUNDS = 100

    BATCH_SIZE = 128
    LR = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1 datapreprocessing
    fld = FederatedLearningDataset(N=K, C=Nc_value)
    user_groups =  fld.non_iid_partition(num_classes_per_client=Nc_value)

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

        #method1 更加保守的解冻策略
        #method2 分层学习率
        #method3 渐进式解冻
        # J=4 Nc=50
        unfreeze_round = ROUNDS // 4
        if r == unfreeze_round:
            print(f"\n>>> Round {r + 1}: Unfreezing the backbone for fine-tuning! <<<")
            for param in global_model.backbone.parameters():
                param.requires_grad = True
        current_lr = LR
        if r >= unfreeze_round:
            current_lr = LR * 0.1
        if r >= ROUNDS * 0.6:
            current_lr = LR * 0.01
        if r >= ROUNDS * 0.8:
            current_lr = LR * 0.001




        m = max(int(C * K), 1)
        selected_clients = np.random.choice(range(K), m, replace=False)

        round_train_loss = 0.0
        local_weights = []

        for client_id in selected_clients:
            # directly call fed_iid train function
            w,l= train_local(
                model=copy.deepcopy(global_model),
                dataset_indices=user_groups[client_id],
                full_dataset=fld.train_dataset,
                J=J_value,
                batch_size=BATCH_SIZE,
                lr=current_lr,
                momentum=0.9,
                weight_decay=5e-4,
                device=device
            )
            local_weights.append(w)
            round_train_loss +=l
        # directly call fed_iid aggregation function
        global_model.load_state_dict(aggregate_weights(local_weights))

        # evaluate (call centralizedmodel evaluate function)
        test_loss, acc = evaluate(global_model, test_loader, criterion, device)
        best_acc = max(best_acc, acc)


        avg_train_loss = round_train_loss / m
        history['train_loss'].append(float(avg_train_loss))
        history['test_loss'].append(float(test_loss))  # 新增这行
        history['test_acc'].append(float(acc))


        print(f"Round [{r + 1:03d}/{ROUNDS}] | "
              f"LR: {current_lr:.5f} | "
              f"Tr Loss: {avg_train_loss:.4f} | "
              f"Te Loss: {test_loss:.4f} | "
              f"Te Acc: {acc:.2f}% | "
              f"Best: {best_acc:.2f}%")

        if (r + 1) % 10 == 0:
            checkpoint_name = f'non_iid_checkpoint_Nc{Nc_value}_J{J_value}_r{r + 1}.pth'
            save_dir='checkpoint/fed_non_iid/J_8_Nc_50'
            os.makedirs(save_dir, exist_ok=True)  # 自动创建不存在的目录
            save_path=os.path.join(save_dir,checkpoint_name)
            torch.save(global_model.state_dict(), save_path)

    return history


if __name__ == "__main__":
    results= run_fed_non_iid_experiment(50,8)
    filename = f'non_iid_history_Nc{50}_J8.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)