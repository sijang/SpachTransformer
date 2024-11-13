import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # Import tqdm for progress bars
from model.SpachTransformer  import SpachTransformer

# Define a dummy dataset (Replace this with your actual dataset)
class CustomDataset(Dataset):
    def __init__(self):
        # Initialize your dataset here
        pass
    
    def __len__(self):
        # Return the number of samples in the dataset
        return 1000
    
    def __getitem__(self, idx):
        # Return a single data sample
        input_data = torch.rand(1, 48, 48, 48)  # Example input data
        target_data = torch.rand(1, 48, 48, 48)  # Example target data
        return input_data, target_data

# Load dataset
train_dataset = CustomDataset()
val_dataset = CustomDataset()

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize the model (using SpachTransformer or Restormer)
model = SpachTransformer()  # Or model = Restormer()
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define loss function and optimizer
criterion = nn.L1Loss()  # Example loss function, adjust as needed
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training settings
num_epochs = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training and validation loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    train_loss = 0.0

    # Wrap the train_loader with tqdm for progress tracking
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

            train_loss += loss.item() * inputs.size(0)  # Accumulate batch loss
            
            # Update tqdm description with running loss
            tepoch.set_postfix(loss=loss.item())

    # Calculate average training loss for the epoch
    train_loss /= len(train_loader.dataset)
    
    # Validation loop with tqdm
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        with tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", unit="batch") as vepoch:
            for inputs, targets in vepoch:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                
                # Update tqdm description with running validation loss
                vepoch.set_postfix(loss=loss.item())

    # Calculate average validation loss for the epoch
    val_loss /= len(val_loader.dataset)
    
    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# # Save the trained model
# torch.save(model.state_dict(), "trained_model.pth")
