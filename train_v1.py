import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

from train import Net, read_series, do_resize_and_center

class LumbarSpineDataset(Dataset):
    def __init__(self, data_dir, df, transform=None):
        self.data_dir = data_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row['study_id']
        series_id = row['series_id']
        series_description = row['series_description']

        volume, dicom_df, _ = read_series(study_id, series_id, series_description)
        image = np.ascontiguousarray(volume.transpose(1, 2, 0))
        image, scale_param = do_resize_and_center(image, reference_size=320)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))

        # Assuming you have ground truth data in your DataFrame
        xy = row['xy']
        z = row['z']
        grade = row['grade']
        zxy_mask = row['zxy_mask']

        sample = {
            'D': [len(image)],
            'image': torch.from_numpy(image).byte(),
            'xy': torch.from_numpy(xy).float(),
            'z': torch.from_numpy(z).long(),
            'grade': torch.from_numpy(grade).long(),
            'zxy_mask': torch.from_numpy(zxy_mask).float()
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        outputs = model(batch)
        
        loss = outputs['zxy_mask_loss'] + outputs['zxy_loss'] + outputs['grade_loss']
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            outputs = model(batch)
            
            loss = outputs['zxy_mask_loss'] + outputs['zxy_loss'] + outputs['grade_loss']
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    # Hyperparameters
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load your data
    data_dir = '/home/ai/neo/data/rsna-2024-lumbar-spine-degenerative-classification'
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))  # Assuming you have a CSV with metadata

    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    train_dataset = LumbarSpineDataset(data_dir, train_df)
    val_dataset = LumbarSpineDataset(data_dir, val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the model
    model = Net(pretrained=True).to(device)
    # 将模型包装为DataParallel
    model = nn.DataParallel(model)
    model.to(device)


    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'model_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    main()