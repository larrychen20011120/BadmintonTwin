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
gmm_c = 5

def gmm_sample(params, gmm_components, max_std=1):
    batch_size = params.size(0)
    params = params.view(batch_size, gmm_components, 5)
    
    means = params[:, :, :2]  # Mean (x, y)
    log_vars = params[:, :, 2:4]  # Log variances
    weights = params[:, :, 4]  # Mixture weights
    
    weights = F.softmax(weights, dim=1)
    component_indices = torch.multinomial(weights, num_samples=1).squeeze(1)
    
    selected_means = torch.gather(means, 1, component_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)).squeeze(1)
    selected_log_vars = torch.gather(log_vars, 1, component_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)).squeeze(1)
    
    stds = torch.clamp(torch.exp(0.5 * selected_log_vars), max=max_std)
    samples = selected_means + stds * torch.randn_like(selected_means)
    
    return samples

def gmm_loss(params, targets, gmm_components, reg_weight=1e-1, center_penalty_weight=1e-1):
    batch_size = targets.size(0)
    params = params.view(batch_size, gmm_components, 5)
    
    means = params[:, :, :2]
    log_vars = params[:, :, 2:4]
    weights = params[:, :, 4]
    
    weights = F.softmax(weights, dim=1)
    targets = targets.unsqueeze(1)
    log_prob = -0.5 * (torch.sum(((targets - means) ** 2) / log_vars.exp(), dim=2) + torch.sum(log_vars, dim=2))
    log_weights = F.log_softmax(weights, dim=1)
    log_prob += log_weights
    
    nll = -torch.logsumexp(log_prob, dim=1).mean()
    
    reg_loss = 0
    for i in range(gmm_components):
        for j in range(i + 1, gmm_components):
            pairwise_distances = torch.norm(means[:, i, :] - means[:, j, :], dim=1)
            reg_loss += torch.clamp(1.0 / pairwise_distances, max=10).mean()
    
    reg_loss = reg_loss / (gmm_components * (gmm_components - 1) / 2)
    center_loss = torch.norm(means, dim=2).reciprocal().mean()
    
    total_loss = nll + reg_weight * reg_loss + center_penalty_weight * center_loss
    return total_loss, nll, reg_loss + center_loss

class SimpleNN(nn.Module):
    def __init__(self, gmm_components):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.shot_type = 0
        self.predict_land_area = nn.Linear(8, gmm_components * 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        land_param = self.predict_land_area(x)
        return land_param
    
    def get_data(self, file_path, shot_type=None,player = None):
        feature_columns = ['player_location_x', 'player_location_y', 'hit_x', 'hit_y', 'opponent_location_x', 'opponent_location_y', 'pre_type', 'type']
        label_columns = ['landing_x', 'landing_y']
        
        data = pd.read_csv(file_path)

        if player != None:
            data = data[data['player'] == player]

        shot_type = self.shot_type
        if shot_type is not None:
            data = data[data['type'] == shot_type]
        
        features = data[feature_columns].values
        labels = data[label_columns].values
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def evaluate_model_by_shot_type(self, file_path,X_test,y_test,epoch=None,writer=None):
        """Evaluate the model using MSE and KL Divergence for each shot type."""
        self.eval()
        shot_types = X_test[:, -1].int().tolist()  # Assuming 'type' is the last feature
        unique_types = sorted(set(shot_types))

        # Initialize dictionaries to store results
        results = {shot_type: {'mse_land': [], 'mse_move': [], 'kl_land': [], 'kl_move': []} for shot_type in unique_types}

        with torch.no_grad():
            # Forward pass for all test samples
            predictions_land_param = self(X_test)

            predictions_land = gmm_sample(predictions_land_param.view(-1, gmm_c * 5), gmm_c, max_std=0.15).unsqueeze(1)

            true_land = y_test[:, 0:2]

            x_min, x_max = -1, 1
            y_min, y_max = -1, 1
            grid_x, grid_y = [5,5] 

            total_land_mse = 0
            total_land_kl = 0

            for shot_type in unique_types:
                indices = [i for i, s in enumerate(shot_types) if s == shot_type]

                if len(indices) == 0:
                    continue

                pred_land_filtered = predictions_land[indices].cpu().numpy()
                true_land_filtered = true_land[indices].cpu().numpy()

                mse_land = F.mse_loss(torch.tensor(pred_land_filtered.squeeze(1)), torch.tensor(true_land_filtered)).item()

                results[shot_type]['mse_land'].append(mse_land)

                x_pred_land = pred_land_filtered[:,:, 0].flatten()
                y_pred_land = pred_land_filtered[:,:, 1].flatten()
                x_true_land = true_land_filtered[:, 0]
                y_true_land = true_land_filtered[:, 1]

                hist_pred_land, _, _ = np.histogram2d(x_pred_land, y_pred_land, bins=[grid_x, grid_y], range=[[x_min, x_max], [y_min, y_max]])
                hist_true_land, _, _ = np.histogram2d(x_true_land, y_true_land, bins=[grid_x, grid_y], range=[[x_min, x_max], [y_min, y_max]])

                pred_dist_land = hist_pred_land / np.sum(hist_pred_land)
                true_dist_land = hist_true_land / np.sum(hist_true_land)

                epsilon = 1e-8
                pred_dist_land += epsilon
                true_dist_land += epsilon

                kl_land = np.sum(pred_dist_land * np.log(pred_dist_land / true_dist_land))

                results[shot_type]['kl_land'].append(kl_land)
                
                total_land_mse += mse_land
                total_land_kl += kl_land

            if (writer is not None) and (epoch is not None):
                writer.add_scalar(f'Metrics/MSE_Landing', total_land_mse, epoch)
                writer.add_scalar(f'Metrics/KL_Landing', total_land_kl, epoch)
                total_land_mse = 0
                total_land_kl = 0

        # Print average metrics for each shot type
        if epoch is None:
            print("\n Evaluation Results by Shot Type:")
            for shot_type, metrics in results.items():
                avg_mse_land = sum(metrics['mse_land']) / len(metrics['mse_land'])
                avg_kl_land = sum(metrics['kl_land']) / len(metrics['kl_land'])

                print(f"\n Shot Type: {shot_type}")
                print(f"MSE (Landing): {avg_mse_land:.6f}")
                print(f"KL Divergence (Landing): {avg_kl_land:.6f}")

                indices = [i for i, s in enumerate(shot_types) if s == shot_type]
                if len(indices) == 0: continue 
                pred_land_filtered = predictions_land[indices].cpu().numpy()
                print(f"len: {len(pred_land_filtered)}")


    def train_model(self, file_path, shot_type=None, epochs=500, batch_size=8, learning_rate=0.0001, save_path='model_weights_gmm71.pth',player = None):
        self.shot_type = shot_type
        save_path=f'model_weights_gmm7{self.shot_type}.pth'
        if player != None:save_path=f'BadmintonEnv/Agent/weight/{player}/model_weights_gmm7{self.shot_type}.pth'

        features, labels = self.get_data(file_path, shot_type,player)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        writer = SummaryWriter(log_dir=f'./runs/SimpleNN_gmm7{self.shot_type}')
        scheduler = StepLR(optimizer, step_size=50, gamma=0.58)
        min_loss = 9999
        
        for epoch in tqdm(range(epochs), desc="Training Progress"):  
            permutation = torch.randperm(X_train.size()[0])
            total_loss = 0
            for i in tqdm(range(0, X_train.size()[0], batch_size), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                indices = permutation[i:i + batch_size]
                batch_inputs, batch_labels = X_train[indices], y_train[indices]
                
                land_params = self(batch_inputs)
                land_loss, _, _ = gmm_loss(land_params.view(-1, gmm_c * 5), batch_labels, gmm_c, reg_weight=0,center_penalty_weight=0)
                
                loss = land_loss
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            avg_loss = total_loss / (X_train.size()[0] / batch_size)
            writer.add_scalar('Loss/train', avg_loss, epoch)

            if (epoch + 1) % 10 == 0:
                    tqdm.write(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f}")
                    self.evaluate_model_by_shot_type(file_path,X_test,y_test,epoch,writer)
                    if avg_loss < min_loss:
                        min_loss = avg_loss
                        torch.save(self.state_dict(), f'BadmintonEnv/Agent/weight/{player}/model_weights_gmm7{self.shot_type}_min.pth')

        # Evaluate on test set
        # with torch.no_grad():
        self.evaluate_model_by_shot_type(file_path,X_test,y_test,epoch=None,writer=None)

        torch.save(self.state_dict(), save_path)
        tqdm.write(f"Training completed. Model weights saved to {save_path}")
        writer.close()

if __name__ == "__main__":
    player = 'Viktor AXELSEN'
    for i in range(8):
        model = SimpleNN(gmm_c)
        shot_type = i +2
        model.train_model('input_data/2_dataset_merge.csv', shot_type=shot_type,player = player)
