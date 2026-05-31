import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from datapreprocessing import FederatedLearningDataset


def test_partition():
    """Test IID and Non-IID data partitioning"""

    print("=" * 50)
    print("Initializing FederatedLearningDataset")
    print("=" * 50)

    N = 10  # 10 clients
    C = 2  # 2 classes per client by default
    fld = FederatedLearningDataset(N=N, C=C)

    print(f"Training set size: {len(fld.train_dataset)}")
    print(f"Validation set size: {len(fld.val_dataset)}")
    print(f"Test set size: {len(fld.test_dataset)}")
    print(f"Training labels shape: {fld.train_targets.shape}")
    print()

    # ========== 1. Test IID Partition ==========
    print("=" * 50)
    print("1. Testing IID Partition")
    print("=" * 50)

    iid_clients = fld.iid_partition()

    print(f"Number of clients: {len(iid_clients)}")
    print(f"Average samples per client: {np.mean([len(v) for v in iid_clients.values()]):.1f}")
    print(f"Std dev of samples per client: {np.std([len(v) for v in iid_clients.values()]):.1f}")
    print()

    print("IID Partition - Label distribution per client:")
    for client_id in range(min(3, N)):  # 仅展示前3个防止刷屏
        indices = list(iid_clients[client_id])
        labels = fld.train_targets[indices]
        label_counts = Counter(labels)
        print(f"  Client {client_id}: {len(indices)} samples, number of classes: {len(label_counts)}")
        print(f"  Class Distribution (Top 10): {dict(list(label_counts.most_common(10)))}")

    # ========== 2. Test Non-IID Partition ==========
    print("\n" + "=" * 60)
    print("2. Testing Non-IID Partition")
    print("=" * 60)

    # 为了防止方案A在循环中把单实例的数据池抽干，每次动态测试都使用独立的清洗实例
    test_cases = [1, 2, 5, 10]

    for Nc in test_cases:
        print(f"\n--- Each client has {Nc} classes ---")

        # 核心修复：每次测试不同 Nc 时实例化一个干净的数据集
        fld_test = FederatedLearningDataset(N=N, C=C)
        non_iid_clients = fld_test.non_iid_partition(num_classes_per_client=Nc)

        print(f"Number of clients: {len(non_iid_clients)}")
        sizes = [len(v) for v in non_iid_clients.values()]
        print(f"Samples per client: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
        print(f"Sample count std dev: {np.std(sizes):.1f}")

        print("\nClass distribution per client:")
        for client_id in range(min(3, N)):
            indices = non_iid_clients[client_id]
            labels = fld_test.train_targets[indices]
            unique_labels = np.unique(labels)
            print(f"  Client {client_id}: {len(indices)} samples, actual classes: {len(unique_labels)} (target: {Nc})")
            print(f"    Contains classes: {sorted(unique_labels)[:10]}")

        all_indices = np.concatenate(list(non_iid_clients.values()))
        all_labels = fld_test.train_targets[all_indices]
        global_classes = len(np.unique(all_labels))
        print(f"Global class coverage: {global_classes}/100")

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

    Nc_visual = 10  # 修复：设为 2 能最清晰地看出 Bug 是否被修复
    print(f"Visualizing Non-IID partition (Nc={Nc_visual})...")

    fld_visual = FederatedLearningDataset(N=N, C=C)
    iid_clients_full = fld_visual.iid_partition()
    non_iid_clients_full = fld_visual.non_iid_partition(num_classes_per_client=Nc_visual)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. IID: 样本量分布
    ax1 = axes[0, 0]
    iid_sizes = [len(v) for v in iid_clients_full.values()]
    ax1.bar(range(N), iid_sizes)
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title(f'IID Partition - Samples per Client (std={np.std(iid_sizes):.1f})')
    ax1.axhline(y=np.mean(iid_sizes), color='r', linestyle='--')

    # 2. Non-IID: 样本量分布
    ax2 = axes[0, 1]
    non_iid_sizes = [len(v) for v in non_iid_clients_full.values()]
    ax2.bar(range(N), non_iid_sizes)
    ax2.set_xlabel('Client ID')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title(f'Non-IID Partition (Nc={Nc_visual}) - Samples per Client (std={np.std(non_iid_sizes):.1f})')
    ax2.axhline(y=np.mean(non_iid_sizes), color='r', linestyle='--')

    # 3. 核心修复：IID 全类别对齐排列 (显示全100个类，拒绝无序模糊)
    ax3 = axes[1, 0]
    sample_client = 0
    iid_indices = list(iid_clients_full[sample_client])
    iid_labels = fld_visual.train_targets[iid_indices]

    # 建立一个完整的 0-99 计数数组
    iid_counts = np.zeros(100)
    for l in iid_labels:
        iid_counts[l] += 1

    ax3.bar(range(100), iid_counts, width=0.8)
    ax3.set_xlabel('Class ID (0-99)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'IID - Client {sample_client} Full Class Distribution')
    ax3.set_xlim(-1, 100)

    # 4. 核心修复：Non-IID 全类别对齐排列
    ax4 = axes[1, 1]
    non_iid_indices = non_iid_clients_full[sample_client]
    non_iid_labels = fld_visual.train_targets[non_iid_indices]

    non_iid_counts = np.zeros(100)
    for l in non_iid_labels:
        non_iid_counts[l] += 1

    ax4.bar(range(100), non_iid_counts, width=0.8)
    ax4.set_xlabel('Class ID (0-99)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Non-IID (Nc={Nc_visual}) - Client {sample_client} Full Class Distribution')
    ax4.set_xlim(-1, 100)

    plt.tight_layout()
    plt.savefig('partition_comparison.png', dpi=150)
    plt.show()

    # ========== 4. Class Coverage Heatmap ==========
    print("\n" + "=" * 60)
    print("4. Class Coverage Heatmap")
    print("=" * 60)

    nc_values = [1, 2, 5, 10, 20]
    fig, axes = plt.subplots(1, len(nc_values), figsize=(20, 4))

    for idx, Nc_test in enumerate(nc_values):
        fld_heatmap = FederatedLearningDataset(N=N, C=C)
        non_iid_clients_test = fld_heatmap.non_iid_partition(num_classes_per_client=Nc_test)

        client_class_matrix = np.zeros((N, 100))
        for client_id in range(N):
            indices = non_iid_clients_test[client_id]
            if len(indices) > 0:
                labels = fld_heatmap.train_targets[indices]
                unique, counts = np.unique(labels, return_counts=True)
                client_class_matrix[client_id, unique] = counts

        ax = axes[idx]
        ax.imshow(client_class_matrix > 0, aspect='auto', cmap='Blues', interpolation='nearest')
        ax.set_xlabel('Class ID')
        ax.set_ylabel('Client ID')
        ax.set_title(f'Nc={Nc_test}')

        coverage = np.mean(np.sum(client_class_matrix > 0, axis=1))
        ax.text(0.5, -0.2, f'Avg classes: {coverage:.1f}', transform=ax.transAxes, ha='center')

    plt.tight_layout()
    plt.savefig('class_coverage_heatmap.png', dpi=150)
    plt.show()

    return iid_clients, non_iid_clients_full


if __name__ == "__main__":
    iid_data, non_iid_data = test_partition()