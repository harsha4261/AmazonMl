import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import timm # PyTorch Image Models
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import re
import requests
from pathlib import Path

# --- 1. Configuration ---
class CFG:
    # General
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    data_dir = Path("dataset")
    image_dir = Path("images")
    train_csv_path = data_dir / "train.csv"
    test_csv_path = data_dir / "test.csv"
    train_img_path = image_dir / "train"
    test_img_path = image_dir / "test"
    
    # Data
    img_size = 224
    
    # Model Hyperparameters
    text_model_name = "distilbert-base-uncased"
    image_model_name = "efficientnet_b0"
    
    # Training Hyperparameters
    batch_size = 32
    epochs = 5 # Start with a few epochs, increase for better performance
    lr = 1e-4
    weight_decay = 1e-6
    patience = 2 # For early stopping

# --- 2. Utility Functions ---
def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def download_image(url, save_path):
    """Downloads an image from a URL and saves it."""
    if not save_path.exists():
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
            else:
                Image.new('RGB', (CFG.img_size, CFG.img_size)).save(save_path)
        except Exception:
            Image.new('RGB', (CFG.img_size, CFG.img_size)).save(save_path)

def extract_ipq(text):
    """Extracts Item Pack Quantity (IPQ) from text."""
    match = re.search(r'Item Pack Quantity: (\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1 # Default to 1 if not found

def calculate_smape(y_true, y_pred):
    """Calculates SMAPE score."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Add a small epsilon to avoid division by zero
    return np.mean(numerator / (denominator + 1e-8)) * 100

# --- 3. Dataset Class ---
class ProductDataset(Dataset):
    def __init__(self, df, tokenizer, image_path, transforms, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.image_path = image_path
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text processing
        text = str(row['catalog_content'])
        ipq = extract_ipq(text)
        
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # --- IMAGE PROCESSING (Corrected Logic) ---

        # **RECOMMENDED APPROACH:** Use sample_id as the filename.
        # This is robust and avoids issues with duplicate image URLs.
        # The download loop below is set up to save files this way.
        img_filename = f"{row['sample_id']}.jpg"
        img_full_path = self.image_path / img_filename

        # **ALTERNATE APPROACH:** If your images are already downloaded and named
        # using the original filename from the URL, uncomment the lines below
        # and comment out the two lines above.
        #
        # original_filename = row['image_link'].split('/')[-1]
        # img_full_path = self.image_path / original_filename
        
        try:
            image = Image.open(img_full_path).convert('RGB')
        except (IOError, FileNotFoundError):
            # Use a placeholder if image is missing/corrupt
            image = Image.new('RGB', (CFG.img_size, CFG.img_size))

        image = self.transforms(image)
        
        # Prepare item
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'image': image,
            'ipq': torch.tensor(ipq, dtype=torch.float32)
        }
        
        # Target processing
        if not self.is_test:
            price = row['price']
            # Log transform the target for stability
            item['target'] = torch.tensor(np.log1p(price), dtype=torch.float32)
            
        return item

# --- 4. Model Architecture ---
class ProductPriceModel(nn.Module):
    def __init__(self, text_model_name, image_model_name):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        text_features_dim = self.text_model.config.hidden_size
        
        self.image_model = timm.create_model(image_model_name, pretrained=True, num_classes=0)
        image_features_dim = self.image_model.num_features
        
        ipq_dim = 1
        
        combined_features_dim = text_features_dim + image_features_dim + ipq_dim
        self.regressor = nn.Sequential(
            nn.Linear(combined_features_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask, image, ipq):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]
        
        image_features = self.image_model(image)
        ipq = ipq.unsqueeze(1)
        
        combined_features = torch.cat([text_features, image_features, ipq], dim=1)
        
        output = self.regressor(combined_features)
        return output

# --- 5. Training and Validation Loops ---
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image = batch['image'].to(device)
        ipq = batch['ipq'].to(device)
        targets = batch['target'].to(device)
        
        outputs = model(input_ids, attention_mask, image, ipq)
        loss = criterion(outputs.squeeze(), targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            ipq = batch['ipq'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(input_ids, attention_mask, image, ipq)
            loss = criterion(outputs.squeeze(), targets)
            
            total_loss += loss.item()
            
            all_preds.extend(np.expm1(outputs.squeeze().cpu().numpy()))
            all_targets.extend(np.expm1(targets.cpu().numpy()))
            
    avg_loss = total_loss / len(dataloader)
    smape_score = calculate_smape(np.array(all_targets), np.array(all_preds))
    
    return avg_loss, smape_score

# --- Main Execution ---
if __name__ == '__main__':
    print(f"Using device: {CFG.device}")
    set_seed(CFG.seed)

    # 1. Prepare Data
    print("Preparing data...")
    df_train = pd.read_csv(CFG.train_csv_path)
    df_test = pd.read_csv(CFG.test_csv_path)

    CFG.train_img_path.mkdir(parents=True, exist_ok=True)
    CFG.test_img_path.mkdir(parents=True, exist_ok=True)

    print("Downloading training images...")
    for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Train Imgs"):
        # Save image with its unique sample_id
        save_path = CFG.train_img_path / f"{row['sample_id']}.jpg"
        download_image(row['image_link'], save_path)
        
    print("Downloading test images...")
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Test Imgs"):
        # Save image with its unique sample_id
        save_path = CFG.test_img_path / f"{row['sample_id']}.jpg"
        download_image(row['image_link'], save_path)
        
    train_df, val_df = train_test_split(df_train, test_size=0.1, random_state=CFG.seed)
    
    # 2. Setup Dataloaders
    print("Setting up dataloaders...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)
    
    data_transforms = transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = ProductDataset(train_df, tokenizer, CFG.train_img_path, data_transforms)
    val_dataset = ProductDataset(val_df, tokenizer, CFG.train_img_path, data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Initialize Model and Training components
    print("Initializing model...")
    model = ProductPriceModel(CFG.text_model_name, CFG.image_model_name).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    criterion = nn.MSELoss()

    # 4. Training Loop
    print("Starting training...")
    best_smape = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG.device)
        val_loss, val_smape = validate_one_epoch(model, val_loader, criterion, CFG.device)
        
        print(f"Epoch {epoch+1}/{CFG.epochs} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SMAPE: {val_smape:.4f}")
        
        if val_smape < best_smape:
            best_smape = val_smape
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ New best model saved with SMAPE: {best_smape:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CFG.patience:
                print(f"Early stopping triggered after {CFG.patience} epochs with no improvement.")
                break

    # 5. Inference and Submission
    print("Starting inference on test set...")
    model.load_state_dict(torch.load("best_model.pth"))
    
    test_dataset = ProductDataset(df_test, tokenizer, CFG.test_img_path, data_transforms, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)
    
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)
            image = batch['image'].to(CFG.device)
            ipq = batch['ipq'].to(CFG.device)
            
            outputs = model(input_ids, attention_mask, image, ipq)
            predictions = np.expm1(outputs.squeeze().cpu().numpy())
            all_predictions.extend(predictions)
            
    submission_df = pd.DataFrame({
        'sample_id': df_test['sample_id'],
        'price': all_predictions
    })
    submission_df['price'] = submission_df['price'].clip(lower=0)
    submission_df.to_csv("test_out.csv", index=False)
    
    print("✅ Submission file 'test_out.csv' created successfully!")