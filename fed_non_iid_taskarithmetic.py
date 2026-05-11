import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
# Reuse existing modules
from centralizedmodel import CentralizedModel, evaluate
from datapreprocessing import FederatedLearningDataset
from fed_non_iid import aggregate_weights  # Same aggregation logic
from taskarithmetic import SparseSGDM, compute_fisher_sensitivity, calibrate_masks
# Reuse the local training function we defined in the IID case
from fed_iid_taskarithmetic import train_local_task_arithmetic


def run_federated_task_arithmetic_non_iid():
    """
    Main execution pipeline for Federated Learning (Non-IID) with Task Arithmetic.
    Focuses on mitigating weight divergence caused by heterogeneous data.
    """
    # Hyperparameters
    K = 10  # Total clients
    Nc = 0.5  # Partial participation (more realistic for Non-IID)
    ROUNDS = 100  # Non-IID usually requires more rounds to converge
    J = 5  # Local steps
    LR = 0.01
    STRATEGY = 'least_sensitive'
    SPARSITY = 0.05  # Higher sparsity (5%) often helps in extreme Non-IID

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Non-IID Data Preparation (Reuse from datapreprocessing.py)
    # Using Dirichlet distribution (non_iid_partition) to simulate data heterogeneity
    fld = FederatedLearningDataset(N=K, C=Nc)
    user_groups = fld.non_iid_partition()  # beta controls the level of Non-IID
    test_loader = DataLoader(fld.test_dataset, batch_size=64, shuffle=False)

    # 2. Global Model Initialization
    global_model = CentralizedModel(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Federated Task Arithmetic (Non-IID)...")
    print(f"Non-IID Beta: 0.5 | Strategy: {STRATEGY} | Sparsity: {SPARSITY}")

    # 3. Federated Training Loop
    for r in range(ROUNDS):
        local_weights = []
        m = max(int(Nc * K), 1)
        selected_clients = torch.randperm(K)[:m].tolist()

        for client_id in selected_clients:
            # Each client calculates its own Fisher Mask based on its unique Non-IID data
            # This ensures local updates don't conflict too harshly with other clients
            w, _ = train_local_task_arithmetic(
                model=copy.deepcopy(global_model),
                dataset_indices=user_groups[client_id],
                full_dataset=fld.train_dataset,
                J=J,
                batch_size=32,
                lr=LR,
                momentum=0.9,
                weight_decay=5e-4,
                device=device,
                strategy=STRATEGY,
                sparsity_ratio=SPARSITY
            )
            local_weights.append(w)

        # 4. Aggregation
        global_weights = aggregate_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # 5. Periodic Evaluation
        if (r + 1) % 5 == 0:
            _, acc = evaluate(global_model, test_loader, criterion, device)
            print(f"Round [{r + 1}/{ROUNDS}] - Global Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    run_federated_task_arithmetic_non_iid()