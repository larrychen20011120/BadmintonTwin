import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    def get_data(self, file_path):
        feature_columns = [
            'ball_round','rally',
            # 'player_location_x', 'player_location_y',
            'opponent_location_x', 'opponent_location_y',
            'landing_x', 'landing_y', 'type'
        ]
        label_column = 'label'
        data = pd.read_csv(file_path)
        features = data[feature_columns].values
        labels = data[label_column].values.reshape(-1, 1)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def train_model(self, file_path, epochs=3000, batch_size=4, learning_rate=0.001, save_path='model_weights_merge6.pth'):
        # Load data
        features, labels = self.get_data(file_path)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir='./runs/SimpleNN3')

        for epoch in range(epochs):
            permutation = torch.randperm(X_train.size()[0])
            total_loss = 0
            for i in range(0, X_train.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_inputs, batch_labels = X_train[indices], y_train[indices]

                # Forward pass
                
                outputs = self(batch_inputs)
                
                loss = criterion(outputs, batch_labels)
                total_loss += loss.item()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Average loss per epoch
            avg_loss = total_loss / (X_train.size()[0] / batch_size)

            # Log the loss to TensorBoard
            writer.add_scalar('Loss/train', avg_loss, epoch)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
                torch.save(self.state_dict(), save_path)

        # Evaluate on the test set
        with torch.no_grad():
            test_outputs = self(X_test)
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            print(f'Test Accuracy: {accuracy * 100:.2f}%')

            predicted_np = predicted.cpu().numpy()
            y_test_np = y_test.cpu().numpy()

            # Calculate F1 Score
            f1 = f1_score(y_test_np, predicted_np, average='binary')
            print(f'Test F1 Score: {f1:.4f}')

        # Save model weights after training
        torch.save(self.state_dict(), save_path)
        print(f"Training completed. Model weights saved to {save_path}")

        # Close TensorBoard writer
        writer.close()

if __name__ == "__main__":
    model = SimpleNN()
    # model.train_model('/home/adsl-1-3/Documents/Demo/from_gdwang/input_data/Processed_dataset_1.csv')
    model_path = '/home/adsl-1-3/Documents/Demo/from_gdwang/BadmintonEnv/Utils/model_weights_merge6.pth'
    # model.load_state_dict(torch.load(model_path))
    model.train_model('/home/adsl-1-3/Documents/Demo/from_gdwang/input_data/Processed_dataset_merge2.csv')
