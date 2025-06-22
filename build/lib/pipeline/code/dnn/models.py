import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from pipeline.code.dnn.dataloader import create_dataloader
from pipeline.code.dnn.dataloader import generate_matrix

from pathlib import Path
from pipeline.code.dnn.config.logger import get_logger

# # dnn framework forward

class ClassifierModule(nn.Module):

    def __init__(
        self,
        input_units,
        output_units,
        hidden_units1,
        hidden_units2,
        hidden_units3,
        mode = 'Train',
        dropout_prob = 0.1
    ):
        super(ClassifierModule, self).__init__()

        self.model_mode = mode

        self.hidden1 = nn.Linear(input_units, hidden_units1)
        self.hidden2 = nn.Linear(hidden_units1, hidden_units2)
        self.softmax = nn.Softmax(dim=1)
        self.drop= nn.Dropout1d(dropout_prob)
        self.hidden3 = nn.Linear(hidden_units2, hidden_units3)
        self.output = nn.Linear(hidden_units3, output_units)

    def forward(self, X):
        
        X = F.relu(self.hidden1(X))
        X = F.relu(self.hidden2(X))
        X = F.relu(self.hidden3(X))

        if self.model_mode == 'Train':
            Embedding = X
            X = self.drop(X)
            X = self.output(X)
            return X,Embedding
        
        elif self.model_mode == 'Weights':
            X = self.drop(X)
            X = self.output(X)
            return X
        
        elif self.model_mode == 'Transform':
            return X
        
        elif self.model_mode == 'Test':
            X = self.drop(X)
            X = self.output(X)
            return X

def prepare_directories(params, data_path, iteration) -> Path:

    data_label = Path(data_path).stem.split("_")[-1]

    if params.independent_test:
        save_path = params.data_files['independent_s_path']
    else:
        save_path = params.data_files['save_path']
        
    model_dir = Path(save_path) / f"Data_{data_label}/Repeat_{iteration}"
    model_dir.mkdir(parents=True, exist_ok=True)

    return model_dir

def run_epoch(model, loader, optimizer=None, mode="train"):

    """Execute one epoch of training/validation"""

    if mode == "train":
        model.train()
        assert optimizer is not None, "Optimizer required for training"
    else:
        model.eval()
    
    total_loss = 0.0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    for inputs, targets in loader:
        inputs = inputs.cuda().float()
        targets = targets.cuda().long()

        with torch.set_grad_enabled(mode == "train"):
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

        # parameter update
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(loader)
    avg_acc = correct / len(loader.dataset)
    
    return avg_loss, avg_acc


def train_dnn(params, train_data, valid_data, iteration, data_path):

    """Main training function for the model"""

    from analyzer import calculate_rank
    logger = get_logger() 

    model_dir = prepare_directories(params, data_path, iteration)
        
    train_loader = create_dataloader(train_data, params.batch_size, shuffle=True,drop_last=True)
    valid_loader = create_dataloader(valid_data, params.batch_size, shuffle=False,drop_last=True)

    if params.independent_test:

        X, y,_,_,_ = generate_matrix(params.data_files['independent_test_path'], params)
        dataset = np.concatenate((X, y), axis=1)
        test_loader = create_dataloader(dataset, 1, shuffle=False,drop_last=True)
    
    # Initialize model and optimizer
    net = ClassifierModule(
        input_units=params.input_units,
        output_units=params.output_units,
        hidden_units1=params.hidden_units1,
        hidden_units2=params.hidden_units2,
        hidden_units3=params.hidden_units3,
        mode='Train'
    ).cuda()
    
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    
    # Training loop
    best_metrics = {"loss": 4.0,"rank": 20.0}
    history = []
    early_stop_counter = 0
    
    for epoch in range(params.max_epochs):
        
        train_loss, train_acc = run_epoch(net, train_loader, optimizer, mode="train") # Training phase
        valid_loss, valid_acc = run_epoch(net, valid_loader, mode="valid") # Validation phase

        if params.independent_test: 
            test_loss, test_acc = run_epoch(net, test_loader, mode="valid")
            avg_rank = 20

        if not params.independent_test:
            avg_rank = calculate_rank(params=params,model=net,epoch=epoch,model_dir=model_dir) # calculate cross-species rank
        
        if valid_loss <= best_metrics["loss"]:
            torch.save(net.state_dict(), model_dir / "Best_validloss_epoch.pth")
            logger.info(f"Saved best model to {model_dir}/Best_validloss_epoch.pth")
            best_metrics["loss"] = valid_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if avg_rank <= best_metrics["rank"]:
            torch.save(net.state_dict(), model_dir / "Best_rank_epoch.pth")
            logger.info(f"Saved best model to {model_dir}/Best_rank_epoch.pth")
            best_metrics["rank"] = avg_rank
        
        if early_stop_counter >= 5:
            logger.info(f"Early stopping triggered at epoch {epoch}") # Early stopping check
            break
        
        scheduler.step()

        logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f} | Avg Rank: {avg_rank:.4f}")
        
        if not params.independent_test:
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "avg_rank": avg_rank
            })

        else:
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "test_loss": test_loss,
                "test_acc":test_acc
            })
    
    # save_result
    torch.save(net.state_dict(), model_dir / "Last_epoch.pth")
    history = pd.DataFrame(history)
    history.to_csv(model_dir / "loss and rank each epoch.csv")

    return None