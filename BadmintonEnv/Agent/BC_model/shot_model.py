import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 12)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x =(self.fc3(x))
        return x
    
    def get_data(self, file_path,player = None):
        feature_columns = ['player_location_x','hit_x','hit_y', 'player_location_y','opponent_location_x', 'opponent_location_y','pre_type']
        label_columns = ['type']
        
        data = pd.read_csv(file_path)

        if player != None:
            data = data[data['player'] == player]
        
        features = data[feature_columns].values
        labels = data[label_columns].values.reshape(-1, 1)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def train_model(self, file_path, epochs=500, batch_size=8, learning_rate=0.001, save_path='model_weights_gmm_serve.pth',player = None):
        save_path=f'model_weights_gmm_serve.pth'
        if player != None:save_path=f'BadmintonEnv/Agent/weight/{player}/model_weights_gmm_serve.pth'

        features, labels = self.get_data(file_path,player)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        writer = SummaryWriter(log_dir=f'./runs/SimpleNN_gmm_serve')
        scheduler = StepLR(optimizer, step_size=50, gamma=0.58)
        min_loss = 9999
        criterion = nn.CrossEntropyLoss(ignore_index = 0)
        
        for epoch in tqdm(range(epochs), desc="Training Progress"):  
            permutation = torch.randperm(X_train.size()[0])
            total_loss = 0
            for i in tqdm(range(0, X_train.size()[0], batch_size), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                indices = permutation[i:i + batch_size]
                batch_inputs, batch_labels = X_train[indices], y_train[indices]
                batch_labels = (batch_labels).squeeze().long()

                
                pred_type = self(batch_inputs)
                
                if batch_labels.dim() == 0:
                    batch_labels = batch_labels.unsqueeze(0)

                loss = criterion(pred_type, batch_labels)

                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / (X_train.size()[0] / batch_size)
            writer.add_scalar('Loss/train', avg_loss, epoch)

            if (epoch + 1) % 10 == 0:
                    tqdm.write(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f}")
                    if avg_loss < min_loss:
                        min_loss = avg_loss
                        torch.save(self.state_dict(), f'BadmintonEnv/Agent/weight/{player}/model_weights_gmm_serve_min.pth')

        torch.save(self.state_dict(), save_path)
        tqdm.write(f"Training completed. Model weights saved to {save_path}")
        writer.close()

if __name__ == "__main__":
    player = 'Kento MOMOTA'
    model = SimpleNN()
    model.train_model('input_data/2_dataset_merge.csv',player = player)

