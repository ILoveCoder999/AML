import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from torch.utils.data import DataLoader, Subset
from centralizedmodel import CentralizedModel, evaluate
from datapreprocessing import FederatedLearningDataset
import json
#Federated_learning independent and identically distributed
def train_local(model, dataset_indices, full_dataset, epochs, batch_size, lr, momentum, weight_decay, device):
    """
    execute the training loop
    J=4 (local steps)，by epoch and batch to achieve。
    """
    #pytorch set model to training model open dropout nad batch normalization
    model.train()
    #dataset_indices extracts a private subset of data belonging to the
    #specific client from full_dataset(global dataset)
    subset = Subset(full_dataset, list(dataset_indices))
    #The client's local data is encapsulated into a data loader
    # with a set batch size (batch_size) and
    # the data shuffled at each epoch (shuffle=True).
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    #The model parameters are updated using a standard stochastic gradient descent (SGD) optimizer,
    #including hyperparameters such as learning rate, momentum, and weight decay (regularization).
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    #a multi-class classification
    criterion = nn.CrossEntropyLoss()

    # J=4 represent 4 local steps
    '''
    To control local computation, to ensure that all clients training in parallel--synchronization
    preventing some clients from overtraining and causing model bias.
    '''
    steps_count = 0
    max_steps = 4
    #training loop
    running_loss = 0.0
    for epoch in range(1): # 通常 J 很小时 1 个 epoch 足够
        for inputs, targets in loader:
            if steps_count >= max_steps:
                break
            inputs, targets = inputs.to(device), targets.to(device) #load data
            optimizer.zero_grad() #clean residual gradients from the previous step
            outputs = model(inputs) #forward propagation :calculate model predictions
            loss = criterion(outputs, targets)#calculate the error between prediction and true labels(loss)
            loss.backward() #backward propagation:calculate gradients
            optimizer.step() #update model weights
            #accumulate the errors generated in these 4 steps
            # making it easier to calculate the average value at the end.
            running_loss += loss.item()
            steps_count += 1
        if steps_count >= max_steps:
            break
            # not return the entire model or the training data, but only the updated model weight parameters (in dictionary format).
            # The central server receives the `state_dict` returned by each client,
            # and then averages and merges them proportionally to generate a new global model.

    return model.state_dict(), running_loss / max_steps

'''
`local_weights_list`:  On the server side, after all selected clients complete their local training 
 they send their updated model weights (`state_dict`) to the server. 
 The server collects these weight dictionaries and places them in this list.
 For example, if 3 clients participated in this round of training,
 this list will contain 3 dictionaries, 
 each containing all the parameters of the entire neural network.
'''
def aggregate_weights(local_weights_list):
    #Deep copy of the first client's tensor,prevent modifying directly original data.
    #avg_weights is a completely independent variable, accumulate the weights from each client.
    avg_weights = copy.deepcopy(local_weights_list[0])
    '''
    Federated aggregation is performed layer by layer. 
    The system first takes the `conv1.weight` of all clients and adds them together,
    then takes the `fc.bias` of all clients and adds them together. 
    '''
    '''
    for key in avg_weights.keys():
    outer loop: `key` refers to the name of each layer in the neural network,
    such as `conv1.weight` (the weights of the first convolutional layer) 
    fc.bias` (the bias of the fully connected layer).
    '''
    '''
    for i in range(1, len(local_weights_list)):`
    (inner loop): Iterates through all remaining clients (starting from the 2nd client, 
    i.e., index 1, because the 1st client is already used as the "base").
    '''
    for key in avg_weights.keys():
        for i in range(1, len(local_weights_list)):
            avg_weights[key] += local_weights_list[i][key]
        #pytorch tensor division function
        #After calculating the sum of the weights of all clients at a certain level (key)
        #divide it by the total number of clients participating in the aggregation, len(local_weights_list).
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights_list))
    #The avg_weights dictionary contains all the average parameters.
    #The server will then load this new dictionary into the global model.
    #then start next communication round
    return avg_weights

def run_fedavg_experiment():
    # configuration parameter
    N = 100           # totol clients of training
    C = 0.1           # Participation rate--randomly selected 10 clients for training in each round.
    J = 4             # local steps
    #The complete process of "distribution -> training -> uploading -> aggregation" needs to be repeated 50 times.
    ROUNDS = 200      # communication round
    BATCH_SIZE = 128
    LR = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data and partition
    fld = FederatedLearningDataset(N=N, C=10)
    user_groups = fld.iid_partition()         # IID partition
    
    # initial global model
    global_model = CentralizedModel(num_classes=100).to(device)
    
    # test set prepare
    test_loader = DataLoader(fld.test_dataset, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting FedAvg: N={N}, C={C}, J={J}")

    history = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': []
    }
    for r in range(ROUNDS):
        if r == 50:
            print(f"\n>>> Round {r + 1}: Unfreezing the backbone for fine-tuning! <<<")
            for param in global_model.backbone.parameters():
                param.requires_grad = True
        current_lr = LR
        if r >= 50:
            current_lr = LR * 0.1  # 第 51 轮 (r=50) 解冻瞬间，立刻降为 0.001
        if r >= 120:
            current_lr = LR * 0.01  # 第 121 轮，降为 0.0001 (进入平稳微调期)
        if r >= 160:
            current_lr = LR * 0.001  # 第 161 轮，降为 0.00001 (收尾阶段)

        local_weights = [] #collect model weight (all participating clients the round)
        m = int(C * N) # selected the number of clients every round
        #select m from N,build []
        #replace=False,prevent drawing the same client repeatedly
        selected_clients = np.random.choice(range(N), m, replace=False)
        
        round_loss = 0.0
        
        # Simulated Parallelism: Sequentially execute local training on the selected client.
        for client_id in selected_clients:
            # Copy current global model parameters
            local_model_dict, loss = train_local(
                model=copy.deepcopy(global_model),#must Otherwise, the model will change after the first client is trained.
                dataset_indices=user_groups[client_id],# Private data assigned to this client
                full_dataset=fld.train_dataset,
                epochs=1,
                batch_size=BATCH_SIZE,
                lr=current_lr,
                momentum=0.9,
                weight_decay=5e-4,
                device=device
            )
            local_weights.append(local_model_dict)
            round_loss += loss
        
        # Aggregate update global model
        global_weights = aggregate_weights(local_weights)
        global_model.load_state_dict(global_weights)

        avg_train_loss = round_loss / m

        # An evaluation is conducted at the end of each round.
        test_loss, test_acc = evaluate(global_model, test_loader, criterion, device)
        print(f"Round [{r + 1}/{ROUNDS}] - Train Loss: {round_loss / m:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        #  将这一轮的数据存入字典
        history['train_loss'].append(float(avg_train_loss))
        history['test_loss'].append(float(test_loss))
        history['test_acc'].append(float(test_acc))
        # checkpoint
        if (r + 1) % 10 == 0:
            torch.save(global_model.state_dict(), f'fedavg_checkpoint_r{r + 1}.pth')

    output_filename = 'fedavg_training_history.json'
    with open(output_filename, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\n>>> 训练完成！所有历史数据已保存至 {output_filename}")


if __name__ == "__main__":
    run_fedavg_experiment()