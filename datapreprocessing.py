import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import  random_split

'''
IID (Independent and Identically Distributed): 
Each client receives uniformly distributed data; 
a 10% validation set is sufficient to represent the global dataset.

Non-IID (Non-Independent and Identically Distributed): 
If a client only has two classes of data (e.g., only "cat" and "dog"), 
then the global validation set is very difficult for it
 because it has never seen the other 98 classes during training.
'''

class FederatedLearningDataset:
    #Build highly standardized data distribution center,Ensure 
    #that each Client receives data in a same format.
    #N represents the number of clients.
    #C represents the number of categories each client can get.
    def __init__(self,N,C):
        self.root="."
        self.ratio=0.1
        self.N=N
        self.C=C
        #Standardization processing, convert to tensors, normalization
        #Original image 32*32 pixels->224*224
        #(Higth,Width,Channels(RGB))
        self.train_transforms=transforms.Compose([
                 transforms.Resize(256),
                 transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机缩放裁剪
                 transforms.RandomHorizontalFlip(p=0.5),
                 transforms.RandomRotation(15),  # 随机旋转
                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
                 transforms.ToTensor(),
                 transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                ]
        )
        self.test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


        #default: training vs test = 50 000 : 10 000,5:1
        #The entire training set, including the training set and the validation set,Random seeds ensure reproducibility
        full_train_dataset=torchvision.datasets.CIFAR100(
            root=self.root,
            train=True,
            download=True,
            transform=self.train_transforms
        ) 

        # test dataset
        # Stored on a central server, it cannot be viewed during training;
        # the final test results are also check.
        self.test_dataset=torchvision.datasets.CIFAR100(
            root=self.root,
            train=False,
            download=True,
            transform=self.test_transforms
        )

        #create train-validation split
        #ration=split ratio use in 0.1 0.2 （Heuristic） 0.1
        val_size=int(len(full_train_dataset)*self.ratio)
        train_size=len(full_train_dataset)-val_size

        #Randomly shuffled according to ratio
        self.train_dataset,self.val_dataset=random_split(
            full_train_dataset,[train_size,val_size],torch.Generator().manual_seed(42)
        )

         # Extract labels for Non-IID partitioning
         # A string of numbers (index) that represents 
         # which samples were assigned to this subset.
         # label 
        self.train_targets = np.array(full_train_dataset.targets)[self.train_dataset.indices]
      
    
    def iid_partition(self):
        print(f"IID partition of {self.N} clients")
        print()
        dataset_per_client=int(len(self.train_dataset)/self.N)
        #Initialize an empty dictionary to store the results.
        #goal:{clientID:[sample index set]}
        dict_client={}
        #Ensure that each client randomly samples
        total_train_dataset_index=list(range(len(self.train_dataset)))
        #ensure original dataset sequence
        np.random.shuffle(total_train_dataset_index)
        #i*dataset_per_client:start position (i+1)*dataset_per_client:end position per client
        # use set ensure Deduplication and find efficient(hash table)
        for i in range(self.N):
            dict_client[i]=set(total_train_dataset_index[i*dataset_per_client:(i+1)*dataset_per_client])
        return dict_client

    #Non-IID data partitioning based on shards. 
    # First, the data group by label 
    # Then, the data is divided into many small pieces (shards).
    # Finally, each client randomly selects Nc shards. 
    # each client only has data from a few classes,  Non-IID (non-independent and identically distributed) data.
    def non_iid_partition(self, num_classes_per_client=None):
        Nc = num_classes_per_client if num_classes_per_client is not None else self.C

        # 1. 找出所有唯一的类别标签
        unique_labels = np.unique(self.train_targets)

        # 2. 将样本索引按类别分组存放
        # class_dict = { class_id: [index1, index2, ...] }
        class_dict = {label: np.where(self.train_targets == label)[0] for label in unique_labels}

        # 每个客户端需要分配到的样本总数
        samples_per_client = len(self.train_dataset) // self.N
        # 在拥有 Nc 个类的情况下，每个类需要贡献多少样本
        samples_per_class_per_client = samples_per_client // Nc

        dict_clients = {i: [] for i in range(self.N)}

        # 3. 为每个客户端随机挑选 Nc 个类别
        for i in range(self.N):
            # 从可用的 100 个类中随机抽 Nc 个类
            selected_classes = np.random.choice(unique_labels, Nc, replace=False)

            client_indices = []
            for cls in selected_classes:
                # 从该类别的样本池中，随机抽取指定数量的样本
                selected_indices = np.random.choice(
                    class_dict[cls],
                    samples_per_class_per_client,
                    replace=True
                )
                client_indices.append(selected_indices)

                # (可选) 如果你不想让不同客户端抽到同一张图，可以把抽走的样本从 pool 中剔除
                # class_dict[cls] = np.setdiff1d(class_dict[cls], selected_indices)

            dict_clients[i] = np.concatenate(client_indices)

        return dict_clients

        
# execute code
if __name__ == "__main__":
    print(f"Pytorch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")

