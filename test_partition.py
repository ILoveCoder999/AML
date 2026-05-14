
import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from datapreprocessing import FederatedLearningDataset


def test_partition():
    """Test IID and Non-IID data partitioning"""

    # Initialize dataset
    print("=" * 50)
    print("Initializing FederatedLearningDataset")
    print("=" * 50)

    N = 10  # 10 clients
    C = 2  # 2 classes per client by default
    fld = FederatedLearningDataset(N=N, C=C)

    print(f"Training set size: {len(fld.train_dataset)}")
    print(f"Validation set size: {len(fld.val_dataset)}")
    print(f"Test set size: {len(fld.test_dataset)}")
    #Training labels shape: (45000,) 0-99
    print(f"Training labels shape: {fld.train_targets.shape}")
    print()

    # ========== 1. Test IID Partition ==========
    print("=" * 50)
    print("1. Testing IID Partition")
    print("=" * 50)

    iid_clients = fld.iid_partition()

    print(f"Number of clients: {len(iid_clients)}")
    #iid_clients.values() get all the data of client,statistic mean and std
    print(f"Average samples per client: {np.mean([len(v) for v in iid_clients.values()]):.1f}")
    print(f"Std dev of samples per client: {np.std([len(v) for v in iid_clients.values()]):.1f}")
    print()

    # Analyze label distribution for IID partition
    print("IID Partition - Label distribution per client:")
    for client_id in range(N):  # Show only first 3 clients
       # [0, 1, 2, ..., 4499]
        indices = list(iid_clients[client_id])
       #corresponding all label
        labels = fld.train_targets[indices]
       #statistic the number of times each category appearing
        label_counts = Counter(labels)

        print(f"  Client {client_id}: {len(indices)} samples, number of classes: {len(label_counts)}")
        print(f"  Class Distribution: {dict(list(label_counts.most_common(10)))}")

    # ========== 2. Test Non-IID Partition ==========
    print("\n" + "=" * 60)
    print("2. Testing Non-IID Partition")
    print("=" * 60)

    # Test different Nc values
    test_cases = [1, 2, 5, 10]

    for Nc in test_cases:
        print(f"\n--- Each client has {Nc} classes ---")

        non_iid_clients = fld.non_iid_partition(num_classes_per_client=Nc)

        # Statistics
        print(f"Number of clients: {len(non_iid_clients)}")
        sizes = [len(v) for v in non_iid_clients.values()]
        print(f"Samples per client: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
        print(f"Sample count std dev: {np.std(sizes):.1f}")

        # Analyze class distribution
        print("\nClass distribution per client:")
        for client_id in range(min(3, N)):
            indices = non_iid_clients[client_id]
            labels = fld.train_targets[indices]
            unique_labels = np.unique(labels)

            # Verify each client has exactly Nc classes
            actual_classes = len(unique_labels)
            print(f"  Client {client_id}: {len(indices)} samples, actual classes: {actual_classes} (target: {Nc})")
            print(f"    Contains classes: {sorted(unique_labels)[:10]}")  # Show first 10 classes

        # Global coverage check
        all_indices = np.concatenate(list(non_iid_clients.values()))
        all_labels = fld.train_targets[all_indices]
        global_classes = len(np.unique(all_labels))
        print(f"\nGlobal class coverage: {global_classes}/100")

        # Check for data overlap between clients
        overlap_found = False
        client_sets = [set(v) for v in non_iid_clients.values()]
        for i in range(len(client_sets)):
            for j in range(i + 1, len(client_sets)):
                if client_sets[i] & client_sets[j]:
                    overlap_found = True
                    break
        print(f"Data overlap between clients: {'Yes' if overlap_found else 'No'}")

    # ========== 3. Visualization Comparison ==========
    print("\n" + "=" * 60)
    print("3. Visualization Comparison")
    print("=" * 60)

    # Select a Non-IID configuration for visualization
    Nc_visual = 2
    print(f"Visualizing Non-IID partition (Nc={Nc_visual})...")

    iid_clients_full = fld.iid_partition()
    non_iid_clients_full = fld.non_iid_partition(num_classes_per_client=Nc_visual)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. IID: Sample count distribution
    ax1 = axes[0, 0]
    iid_sizes = [len(v) for v in iid_clients_full.values()]
    ax1.bar(range(N), iid_sizes)
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title(f'IID Partition - Samples per Client (std={np.std(iid_sizes):.1f})')
    ax1.axhline(y=np.mean(iid_sizes), color='r', linestyle='--', label=f'Mean: {np.mean(iid_sizes):.0f}')
    ax1.legend()

    # 2. Non-IID: Sample count distribution
    ax2 = axes[0, 1]
    non_iid_sizes = [len(v) for v in non_iid_clients_full.values()]
    ax2.bar(range(N), non_iid_sizes)
    ax2.set_xlabel('Client ID')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title(f'Non-IID Partition (Nc={Nc_visual}) - Samples per Client (std={np.std(non_iid_sizes):.1f})')
    ax2.axhline(y=np.mean(non_iid_sizes), color='r', linestyle='--', label=f'Mean: {np.mean(non_iid_sizes):.0f}')
    ax2.legend()

    # 3. IID: Class distribution (select one client)
    ax3 = axes[1, 0]
    sample_client = 0
    iid_indices = list(iid_clients_full[sample_client])
    iid_labels = fld.train_targets[iid_indices]
    iid_counter = Counter(iid_labels)
    classes = list(iid_counter.keys())[:20]  # Show first 20 classes
    counts = [iid_counter[c] for c in classes]
    ax3.bar(range(len(classes)), counts)
    ax3.set_xlabel('Class ID')
    ax3.set_ylabel('Count')
    ax3.set_title(f'IID - Client {sample_client} Class Distribution (Total: {len(iid_counter)} classes)')
    ax3.set_xticks(range(len(classes)))
    ax3.set_xticklabels(classes, rotation=45, fontsize=8)

    # 4. Non-IID: Class distribution (same client)
    ax4 = axes[1, 1]
    non_iid_indices = non_iid_clients_full[sample_client]
    non_iid_labels = fld.train_targets[non_iid_indices]
    non_iid_counter = Counter(non_iid_labels)
    classes_non_iid = list(non_iid_counter.keys())
    counts_non_iid = [non_iid_counter[c] for c in classes_non_iid]
    ax4.bar(range(len(classes_non_iid)), counts_non_iid)
    ax4.set_xlabel('Class ID')
    ax4.set_ylabel('Count')
    ax4.set_title(
        f'Non-IID (Nc={Nc_visual}) - Client {sample_client} Class Distribution (Total: {len(classes_non_iid)} classes)')
    ax4.set_xticks(range(len(classes_non_iid)))
    ax4.set_xticklabels(classes_non_iid, rotation=45, fontsize=8)

    plt.tight_layout()
    plt.savefig('partition_comparison.png', dpi=150)
    plt.show()
    print("Visualization saved as 'partition_comparison.png'")

    # ========== 4. Additional Analysis: Class Coverage Heatmap ==========
    print("\n" + "=" * 60)
    print("4. Class Coverage Heatmap")
    print("=" * 60)

    # Select different Nc values for comparison
    nc_values = [1, 2, 5, 10, 20]

    fig, axes = plt.subplots(1, len(nc_values), figsize=(20, 4))

    for idx, Nc_test in enumerate(nc_values):
        if Nc_test > 100:
            continue

        non_iid_clients_test = fld.non_iid_partition(num_classes_per_client=Nc_test)

        # Create client-class matrix
        client_class_matrix = np.zeros((N, 100))

        for client_id in range(N):
            indices = non_iid_clients_test[client_id]
            labels = fld.train_targets[indices]
            unique, counts = np.unique(labels, return_counts=True)
            client_class_matrix[client_id, unique] = counts

        # Plot heatmap (binary, showing only whether client has the class)
        ax = axes[idx]
        im = ax.imshow(client_class_matrix > 0, aspect='auto', cmap='Blues', interpolation='nearest')
        ax.set_xlabel('Class ID')
        ax.set_ylabel('Client ID')
        ax.set_title(f'Nc={Nc_test}\n(Each client has {Nc_test} classes)')
        ax.set_xticks(range(0, 100, 10))
        ax.set_yticks(range(0, N, 2))

        # Add statistics
        coverage = np.mean(np.sum(client_class_matrix > 0, axis=1))
        ax.text(0.5, -0.15, f'Avg classes/client: {coverage:.1f}',
                transform=ax.transAxes, ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('class_coverage_heatmap.png', dpi=150)
    plt.show()
    print("Heatmap saved as 'class_coverage_heatmap.png'")

    # ========== 5. Summary ==========
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    print("  IID Partition Characteristics:")
    print("   - Each client has samples from all classes")
    print("   - Sample count distribution is uniform")
    print("   - Simulates ideal scenario")

    print("\n  Non-IID Partition Characteristics:")
    print("   - Each client has samples from only a few classes")
    print("   - Sample counts remain relatively balanced")
    print("   - Simulates real-world scenario (limited user interests)")
    print("   - Smaller Nc means more severe Non-IID distribution")

    return iid_clients, non_iid_clients_full


if __name__ == "__main__":
    # Run tests
    iid_data, non_iid_data = test_partition()

    # Additional: Print data integrity checks
    print("\n" + "=" * 60)
    print("Quick Data Integrity Check")
    print("=" * 60)

    fld_quick = FederatedLearningDataset(N=10, C=2)

    # Check if Non-IID partition covers all training samples
    non_iid_test = fld_quick.non_iid_partition(num_classes_per_client=2)
    all_indices = np.concatenate(list(non_iid_test.values()))
    unique_indices = np.unique(all_indices)

    print(f"Total training samples: {len(fld_quick.train_dataset)}")
    print(f"Samples covered by Non-IID partition: {len(unique_indices)}")
    print(f"Coverage ratio: {len(unique_indices) / len(fld_quick.train_dataset) * 100:.2f}%")
    print(f"Missing samples: {'Yes' if len(unique_indices) < len(fld_quick.train_dataset) else 'No'}")

    # Check for data duplication
    print(f"Data duplication: {'Yes' if len(all_indices) > len(unique_indices) else 'No'}")