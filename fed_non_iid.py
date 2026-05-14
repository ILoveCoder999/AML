import torch
import numpy as np
import json
import copy
from torch.utils.data import DataLoader
from centralizedmodel import CentralizedModel, evaluate
from datapreprocessing import FederatedLearningDataset
from fed_iid import train_local, aggregate_weights


def run_experiment(Nc_value, J_value, is_iid=False):

    K = 100
    C = 0.1
    TOTAL_STEPS = 400
    ROUNDS = TOTAL_STEPS // J_value
    BATCH_SIZE = 32
    LR = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1 datapreprocessing
    fld = FederatedLearningDataset(N=K, C=Nc_value)
    user_groups = fld.iid_partition() if is_iid else fld.non_iid_partition()

    # 2 model initialization
    global_model = CentralizedModel(num_classes=100).to(device)
    test_loader = DataLoader(fld.test_dataset, batch_size=64, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    print(f"\n>> Running: {'IID' if is_iid else f'Nc={Nc_value}'}, J={J_value}")

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
            w, _ = train_local(
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
        _, acc = evaluate(global_model, test_loader, criterion, device)
        best_acc = max(best_acc, acc)

        if (r + 1) % 5 == 0:
            print(f"Round {r + 1}/{ROUNDS} | Best Acc: {best_acc:.2f}%")

    return best_acc


if __name__ == "__main__":
    results = {}
    # run combination experiment
    for nc in [1, 5, 10, 50]:
        for j in [4, 8, 16]:
            tag = f"Nc{nc}_J{j}"
            results[tag] = run_experiment(nc, j)

    with open('non_iid_results.json', 'w') as f:
        json.dump(results, f, indent=4)