import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, Subset

# Reusing modules from your provided files
from centralizedmodel import CentralizedModel, evaluate
from datapreprocessing import FederatedLearningDataset
from fed_iid import aggregate_weights  # Standard FedAvg aggregation
from taskarithmetic import SparseSGDM, compute_fisher_sensitivity, calibrate_masks


def train_local_task_arithmetic(model, dataset_indices, full_dataset, J, batch_size, lr,
                                momentum, weight_decay, device, strategy, sparsity_ratio):
    """
    Performs local training on a client using Task Arithmetic masking.
    Only a subset of parameters is updated based on Fisher Information sensitivity.
    """
    model.train()
    # Create a local data subset for the specific client
    subset = Subset(full_dataset, list(dataset_indices))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    # --- Task Arithmetic Core: Sensitivity Analysis ---
    # Calculate Fisher Information sensitivity scores using a small number of batches
    # This identifies which parameters are "critical" for the local task
    sensitivity_scores = compute_fisher_sensitivity(
        model, loader, criterion, device, num_batches=5
    )

    # Generate binary masks (1 for update, 0 for freeze) based on the chosen strategy
    # 'least_sensitive' updates weights that don't destroy pre-trained features
    masks = calibrate_masks(
        model,
        strategy=strategy,
        sparsity_ratio=sparsity_ratio,
        sensitivity_scores=sensitivity_scores
    )

    # Initialize the specialized Sparse Optimizer
    # It ensures that only parameters marked in the mask receive gradient updates
    optimizer = SparseSGDM(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        masks=masks
    )

    # Perform J local SGD steps
    completed_steps = 0
    while completed_steps < J:
        for inputs, targets in loader:
            if completed_steps >= J:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()  # Masks are applied inside the optimizer step

            completed_steps += 1

    return model.state_dict(), loss.item()


def run_federated_task_arithmetic_iid():
    """
    Main execution pipeline for Federated Learning (IID) with Task Arithmetic updates.
    """
    # Hyperparameters
    K = 10  # Total number of clients
    C = 1.0  # Fraction of clients participating per round
    ROUNDS = 50  # Number of communication rounds
    J = 4  # Number of local update steps per client
    BATCH_SIZE = 32
    LR = 0.01
    STRATEGY = 'least_sensitive'
    SPARSITY = 0.1  # Only 10% of the parameters will be updated locally

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data Preparation (Reuse from datapreprocessing.py)
    # Generates IID partitions across 10 clients
    fld = FederatedLearningDataset(N=K, C=10)
    user_groups = fld.iid_partition()
    test_loader = DataLoader(fld.test_dataset, batch_size=64, shuffle=False)

    # 2. Global Model Initialization (Reuse from centralizedmodel.py)
    # Pre-trained ViT-S/16 backbone
    global_model = CentralizedModel(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Federated Task Arithmetic (IID)...")
    print(f"Strategy: {STRATEGY} | Target Sparsity: {SPARSITY}")

    # 3. Federated Training Loop
    for r in range(ROUNDS):
        local_weights = []
        m = max(int(C * K), 1)
        # Randomly select a subset of clients
        selected_clients = torch.randperm(K)[:m].tolist()

        for client_id in selected_clients:
            # Each client performs sparse local fine-tuning
            w, _ = train_local_task_arithmetic(
                model=copy.deepcopy(global_model),
                dataset_indices=user_groups[client_id],
                full_dataset=fld.train_dataset,
                J=J,
                batch_size=BATCH_SIZE,
                lr=LR,
                momentum=0.9,
                weight_decay=5e-4,
                device=device,
                strategy=STRATEGY,
                sparsity_ratio=SPARSITY
            )
            local_weights.append(w)

        # 4. Global Aggregation (Reuse logic from fed_iid.py)
        # Averages the sparse updates from all participating clients
        global_weights = aggregate_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # 5. Periodic Global Evaluation
        _, acc = evaluate(global_model, test_loader, criterion, device)
        print(f"Round [{r + 1}/{ROUNDS}] - Global Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    run_federated_task_arithmetic_iid()