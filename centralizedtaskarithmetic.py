import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Reuse existing components
from centralizedmodel import CentralizedModel, evaluate, learnrated_schedule
from datapreprocessing import FederatedLearningDataset
from taskarithmetic import SparseSGDM, compute_fisher_sensitivity, calibrate_masks


def run_centralized_task_arithmetic(strategy='least_sensitive', sparsity_ratio=0.1):
    """
    Main training pipeline for Centralized Task Arithmetic baseline.
    """
    # 1. Environment & Data Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensuring consistency with FL experiments (CIFAR-100, 224x224 resize)
    fld = FederatedLearningDataset(N=10, C=10)
    train_loader = DataLoader(fld.train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(fld.val_dataset, batch_size=128, shuffle=False)

    # 2. Model Initialization
    # Loads ViT-S/16 backbone pre-trained with DINO
    model = CentralizedModel(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()

    # --- Task Arithmetic Core Logic ---

    # Step A: Sensitivity Calibration
    # Uses a few batches to estimate empirical Fisher Information (gradient squared)
    print(f"Calculating Fisher Information sensitivity...")
    sensitivity_scores = compute_fisher_sensitivity(
        model, train_loader, criterion, device, num_batches=10
    )

    # Step B: Mask Generation
    # Determines which weights to 'freeze' vs 'update' based on the ratio
    print(f"Calibrating masks: Strategy={strategy}, Sparsity={sparsity_ratio}")
    masks = calibrate_masks(
        model,
        strategy=strategy,
        sparsity_ratio=sparsity_ratio,
        sensitivity_scores=sensitivity_scores
    )

    # Step C: Sparse Optimizer Setup
    # SparseSGDM zeros out gradients for parameters where mask == 0
    optimizer = SparseSGDM(
        model.parameters(),
        lr=0.0001,
        momentum=0.9,
        weight_decay=5e-4,
        masks=masks
    )
    # ----------------------------------

    # 3. Training Loop
    epochs = 30
    scheduler = learnrated_schedule(optimizer, warmup_epochs=5, total_epochs=epochs)

    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': []}

    print(f"Starting sparse training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()  # The SparseSGDM applies the masks here

        scheduler.step()

        # Periodic Evaluation
        _, val_acc = evaluate(model, val_loader, criterion, device)
        if val_acc > best_acc:
            best_acc = val_acc

        print(f"Epoch [{epoch + 1}/{epochs}] | Val Acc: {val_acc:.2f}%")
        history['val_acc'].append(val_acc)

    return best_acc, history


if __name__ == "__main__":
    # Example Baseline Run: Updating only the 10% least-sensitive parameters
    final_acc, _ = run_centralized_task_arithmetic(strategy='least_sensitive', sparsity_ratio=0.1)
    print(f"\nFinal Baseline Accuracy (Sparsity 0.1): {final_acc:.2f}%")