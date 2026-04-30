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
        self.transforms=transforms.Compose([
                 transforms.Resize((224,224)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                ]
            )
        #Temporary dataset, not normalized
        temp_dataset = torchvision.datasets.CIFAR100(
            root=self.root, train=True, download=True, transform=self.transforms
        )

        #default: training vs test = 50 000 : 10 000,5:1
        #The entire training set, including the training set and the validation set,Random seeds ensure reproducibility
        full_train_dataset=torchvision.datasets.CIFAR100(
            root=self.root,
            train=True,
            download=True,
            transform=self.transforms
        ) 

        # testdataset 
        # Stored on a central server, it cannot be viewed during training;
        # the final test results are also check.
        self.test_dataset=torchvision.datasets.CIFAR100(
            root=self.root,
            train=False,
            download=True,
            transform=self.transforms
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
        print(f"IID partiton of {self.N} clients")
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
    def non_iid_partition(self):
        print(f"Non-IID partition (Nc={self.C}) for {self.N}")
        '''
        1 order by labels
        '''
        #np -> high efficience  vstack accept np array
        #create train set index(0-1):house number
        total_train_dataset_index=np.arange(len(self.train_dataset))
        #get all sample real label
        total_train_labels=self.train_targets
        #Stack the indices and labels vertically to form a 2 × N matrix. 
        # The first row : indices, and the second row : the corresponding labels.
        total_train_dataset_index_labels=np.vstack((total_train_dataset_index,total_train_labels))
        #matrix slicing Numpy:fancy indexing and vector opertation
        #order by label
        total_train_dataset_index_labels = total_train_dataset_index_labels[:, total_train_dataset_index_labels[1, :].argsort()]
        #Sorted index ,we need to first line
        order_index=total_train_dataset_index_labels[0,:]

        '''
        2. Create Shards 
        '''
        # Total number of shards = (Number of clients) * (Shards per client)
        total_shards = self.N * self.C

        # Number of samples in each shard
        shards_size = int(len(self.train_dataset) / total_shards)

        # 1. Initialize an empty list to store the results
        index_shard = []

        # 2. Start the loop to slice the data into shards
        for i in range(total_shards):
            # Calculate the starting and ending positions for the current shard
            start = i * shards_size
            end = (i + 1) * shards_size
            
            # Slice a portion of the sorted indices (one shard)
            current_shard = order_index[start:end]
            
            # CRITICAL: Append the shard to the list to keep all shards (prevents overwriting)
            index_shard.append(current_shard)
           

        '''
        3. Assign Shards to Clients
        '''
        # Initialize a dictionary to store indices for each client.

        # Client ID:Value: NumPy array of sample indices.

        dict_clients = {i: np.array([], dtype='int64') for i in range(self.N)}
        
        # Create a pool  as available shard indices [0, 1, 2, ..., total_shards-1].
        tmp_shards = list(range(total_shards))
        
        # Iterate through each client to assign data.
        for i in range(self.N):
            # Each client randomly picks 'self.C' shards from the pool.
            for _ in range(self.C):
                # Randomly select one shard index from pool.
                shard_index = np.random.choice(tmp_shards)
                
                # Concatenate the selected shard's data indices into the client's array.
                dict_clients[i] = np.concatenate((dict_clients[i], index_shard[shard_index]))
                
                # Remove the selected shard from the pool to ensure it isn't picked again.
                # This guarantees that each shard is assigned to exactly one client.
                tmp_shards.remove(shard_index)

        #mapping of clients :their respective data indices.
        return dict_clients


        
# 执行测试
if __name__ == "__main__":
    print(f"Pytorch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")

