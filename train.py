import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import SpachTransformer

# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, img_size=48):
        self.img_size = img_size
    
    def __len__(self):
        return 1000  # Define the number of samples in the dataset
    
    def __getitem__(self, idx):
        input_data = torch.rand(1, self.img_size, self.img_size, self.img_size)  # Simulated input data
        target_data = torch.rand(1, self.img_size, self.img_size, self.img_size)  # Simulated target data
        return input_data, target_data

# Define the training function
def train_model(simulated_img_size=128, num_epochs=20, batch_size=1, learning_rate=1e-4):
    # Determine the swin window size based on image size
    swin_window_size = (8, 8, 8) if simulated_img_size == 128 else (6, 6, 6)

    # Load dataset
    train_dataset = CustomDataset(img_size=simulated_img_size)
    val_dataset = CustomDataset(img_size=simulated_img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = SpachTransformer(swin_window_size=swin_window_size, num_blocks=[1, 1, 1, 1])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and validation loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Training loop with progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
            for inputs, targets in tepoch:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate batch loss
                train_loss += loss.item() * inputs.size(0)
                
                # Update tqdm with current loss
                tepoch.set_postfix(loss=loss.item())

        # Calculate average training loss for the epoch
        train_loss /= len(train_loader.dataset)
        
        # Validation loop with progress bar
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", unit="batch") as vepoch:
                for inputs, targets in vepoch:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Accumulate batch loss
                    val_loss += loss.item() * inputs.size(0)
                    
                    # Update tqdm with current validation loss
                    vepoch.set_postfix(loss=loss.item())

        # Calculate average validation loss for the epoch
        val_loss /= len(val_loader.dataset)
        
        # Print epoch progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the trained model
    # torch.save(model.state_dict(), "trained_model.pth")

# Main function to parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpachTransformer model with custom parameters.")
    parser.add_argument("--simulated_img_size", type=int, default=128, help="Size of the input image (default: 128)")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training (default: 20)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training (default: 1)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer (default: 1e-4)")

    args = parser.parse_args()

    # Call the training function with parsed arguments
    train_model(simulated_img_size=args.simulated_img_size,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate)
