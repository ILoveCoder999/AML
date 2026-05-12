import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
from datapreprocessing import FederatedLearningDataset
from torch.utils.data import DataLoader
import itertools
from torch.optim.lr_scheduler import LambdaLR
import os

class CentralizedModel(nn.Module):
    #During training, 10% of neurons are randomly stop
    #Prevent the model from rote overfitting.
    def __init__(self,num_classes=100,dropout_rate=0.1):
        super(CentralizedModel,self).__init__()
        '''
        Our idea:
        use a "backbone" pre-trained on massive amounts of data -> extract features
        add a simple "head" to adapt to  specific task.
        '''
        '''
        ViT: Stands for Vision Transformer, give up cnn, uses a self-attention mechanism.
        S (Small):  a smaller version, balancing speed and accuracy.
        Represents that the image is processed by cutting it into 16x16 pixel patches.
        '''
        #load pre-model as backbone transform
        self.backbone=torch.hub.load('facebookresearch/dino:main','dino_vits16')
        #dino_vits16:fix input -> feature vector of length 384
        self.embed_dimension=384
        #before classifcation,enhance model generalization
        self.dropout=nn.Dropout(dropout_rate)
        #A fully connected layer (linear transformation)
        #features mapping to  100 categories.(highest-scoring)
        self.head=nn.Linear(self.embed_dimension,num_classes)
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    #claculate forward
    def forward(self,x):
        features=self.backbone(x)
        features=self.dropout(features)
        return self.head(features)

'''
learning rate scheduler
'''

def learnrated_schedule(optimizer, warmup_epochs=5, total_epochs=30):
    def learningrate_lambda(epoch):
        # 1. Linear Warmup Phase
        if epoch < warmup_epochs:
            # Gradually increase the multiplier from 0 to 1(initial lr)
            return float(epoch) / float(max(1, warmup_epochs))
        # 2. Cosine Annealing Phase
        else:
            #the decay phase (from 0.0 to 1.0)
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            # 0.5 * (1 + cos) shifts and scales this range to [1.0, 0.0]. cosfunction smoothing
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    # LambdaLR applies the multiplier returned by lr_lambda to the initial LR
    return LambdaLR(optimizer, learningrate_lambda)

def train_one_epoch(model,loader,criterion,optimizer,device):
    #The model uses a Dropout layer. 
    #Calling this method activates Dropout (randomly discarding neurons to prevent overfitting).
    model.train()
    #cumulative loss value
    running_loss=0.0
    #cumlative correct predicted samples number
    correct=0
    #cumlative processed samples
    total=0
    for batch_idx, (inputs, targets) in enumerate(loader):
        #data label -> specific device(gpu)
        inputs, targets = inputs.to(device), targets.to(device)
        '''
        core 5 steps
        '''
        #Clear gradients.
        #clear the gradients from the previous batch before processing a new batch of data.
        optimizer.zero_grad()
        #Forward propagation. 
        #the backbone  extract features,  map to 100 category via the head.
        outputs = model(inputs)
        #caculate loss 
        #Compare the model's predictions with the true labels (targets).
        loss = criterion(outputs, targets)
        #Backpropagation. 
        #Calculate gradient parameter (weight) based on the loss value.
        loss.backward()
        #Update parameters. Using the calculated gradients, adjust the model weights 
        # according to the optimization algorithm (specified by the learning rate scheduler).
        optimizer.step()

        running_loss += loss.item() #loss is tensor -> float
        #output is a tensor of shape (BatchSize, 100),
        #where each row scores across 100 categories.
        # finds the maximum value in dimension 1 
        _, predicted = outputs.max(1) 
        #Number of samples included in the current batch
        #the total number of images that have already been processed.
        total += targets.size(0)
        '''
        The predicted indexes are compared one by one with the actual labels,
         generating a boolean tensor 
         (True if they are the same, False if they are different).
        '''
        #sum(): Counts the number of True values ​​in the tensor.
        #.item():into integer in python
        correct += predicted.eq(targets).sum().item()
    
    #sum(batch_loss)/nums_batches
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

#validationSet,Test set
def evaluate(model, loader, criterion, device):
    #close dropout ensure same input get same output
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    #No gradient we only get result
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def run_centralized_baseline(hyperparameters=None,seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed(seed)
    Device=torch.device("cuda"if torch.cuda.is_available()else "cpu")
    print(f"Using Device:{Device}")
    print(f"seed:{seed}")
    #first train set hyperparameters
    if hyperparameters is None:
        hyperparameters={
            'batch_size': 128,
            'epochs': 50,
            'lr': 0.0001,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'dropout_rate': 0.1,
            'warmup_epochs': 5,
        }
    BATCH_SIZE = hyperparameters['batch_size']
    EPOCHS = hyperparameters['epochs']
    LR = hyperparameters['lr']
    MOMENTUM = hyperparameters['momentum']
    WEIGHT_DECAY = hyperparameters['weight_decay']
    DROPOUT_RATE = hyperparameters['dropout_rate']
    WARMUP_EPOCHS = hyperparameters['warmup_epochs']
    fld = FederatedLearningDataset(N=10, C=2)
    train_dataset = fld.train_dataset   
    val_dataset = fld.val_dataset       
    test_dataset = fld.test_dataset    
    print(f"trainingset: {len(train_dataset)}")
    print(f"validationset: {len(val_dataset)}")
    print(f"testset: {len(test_dataset)}")
    #create data loader shuffle:discorder num_worker:multithread pin_memory:accelerate GPU
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,  shuffle=False, num_workers=4,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,pin_memory=True)
    model=CentralizedModel(num_classes=100,dropout_rate=DROPOUT_RATE).to(Device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                         weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = learnrated_schedule(optimizer, warmup_epochs=WARMUP_EPOCHS,
                               total_epochs=EPOCHS)
    print(f"Starting centralized training for {EPOCHS} epochs...")
    print(f"Hyperparameters: LR={LR}, Batch={BATCH_SIZE}, WD={WEIGHT_DECAY}")
    #start epoch=0,1,2,3...
    i = 0
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(i, EPOCHS):
        # --- 重点：在这里加入解冻逻辑 ---
        if epoch == 20:
            print(">>> Epoch 20: Unfreezing the backbone for fine-tuning...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            
            # 手动调低基础学习率，防止剧烈震荡破坏预训练特征
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
        # -----------------------------
        # Train for one epoch
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion,
                                       optimizer, Device)
        # Validate
        v_loss, v_acc = evaluate(model, val_loader, criterion, Device)
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update learning rate
        scheduler.step()
        # Record history
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.2f}% | LR: {current_lr:.6f}")
        # After calculating v_acc
        '''
        model.state_dict()
        Save the model: Only save the parameters, 
        This results in a very small file.
        '''
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), 'best_centralized_model.pth')
            print(f"Saved new best model with validation accuracy: {v_acc:.2f}%")
    # Test on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, Device)   
    print(f"Test Accuracy: {test_acc:.2f}%")

    
    return model, history, best_val_acc


def hyperparameter_search(seed=42):
    """
    Grid Search:Find best hyperparameters。
    Iterate through the defined parameter space
    return the parameter combination
    that performs best on the test set.
    """
    print("=" * 50)
    print("Start Hyperparameter Grid Search:")
    print("=" * 50)


    search_space = {
        'lr': [0.001, 0.0001],
        'batch_size': [64, 128],
        'weight_decay': [1e-4, 5e-4]
    }

    base_hyperparameters = {
        'epochs': 30,
        'momentum': 0.9,
        'dropout_rate': 0.1,
        'warmup_epochs': 3,
    }

    # Cartesian product generates all possible combinations
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_acc = 0.0
    best_params = None
    output_dir = './hyperparameter_search_results'
    os.makedirs(output_dir, exist_ok=True)
    results_log=[]

    # 4. Train by traversing all combinations
    for i, combo in enumerate(combinations):
        print(f"\n[Search {i + 1}/{len(combinations)}] Testing combinations: {combo}")

        # combination dynamic parameters and static parameters
        current_hp = base_hyperparameters.copy()
        current_hp.update(combo)

        # run training pipeline
        _, _, val_acc = run_centralized_baseline(hyperparameters=current_hp, seed=seed)

        # record best result
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = current_hp.copy()

    print("\n" + "=" * 50)
    print("hyperparameter search finished")
    print(f"best hyperparameter accuracy rate:{best_acc:.2f}%")
    print(f"best hyperparameter combination: {best_params}")
    print("=" * 50)

    # --- SAVE TO FILES ---
    # 1. Save all trial history
    history_path = os.path.join(output_dir, 'search_history.json')
    with open(history_path, 'w') as f:
        json.dump(results_log, f, indent=4)

    # 2. Save only the best parameters
    best_path = os.path.join(output_dir, 'best_hyperparameters.json')
    with open(best_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    # --- FINAL SUMMARY TABLE ---
    print("\n" + "=" * 70)
    print(f"{'Trial':<8} | {'LR':<10} | {'Batch':<8} | {'WD':<10} | {'Val Acc (%)':<12}")
    print("-" * 70)
    for res in results_log:
        print(
            f"{res['trial']:<8} | {res['lr']:<10.4f} | {res['batch_size']:<8} | {res['weight_decay']:<10.1e} | {res['val_acc']:<12.2f}")

    print("-" * 70)
    print(f" Search complete. Results saved to '{output_dir}/'")
    print(f" Best Accuracy: {best_acc:.2f}%")
    print("=" * 70 + "\n")
    return best_params

def plot_training_curves(history):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    #loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', color='blue')
    axes[1].plot(history['val_acc'], label='Val Acc', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.show()
    print("The training curve has been saved as 'training_curves.png'")


if __name__ == "__main__":

    print("---hint:if you need to repeat hyperparameter search,please cancel below code comment")
    best_params = hyperparameter_search(seed=42)
    print("Please update best hyperparameters above output  into optimal_hyperparameters")
    optimal_hyperparameters = best_params

    optimal_hyperparameters = {
        'batch_size': 128,
        'epochs': 30,          
        'lr': 0.0001,         
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'dropout_rate': 0.1,
        'warmup_epochs': 5,
    }
    
    model, history, test_acc = run_centralized_baseline(hyperparameters=optimal_hyperparameters, seed=42)
    
    
    with open('training_history.json', 'w') as f:
        # numpy->Python type
        json_history = {k: [float(x) for x in v] for k, v in history.items()}
        json.dump(json_history, f, indent=2)
    
    print(f"\ntrain_history 'training_history.json'")
  
    print(f"best_centralized_model'best_centralized_model.pth'")
    plot_training_curves(history)