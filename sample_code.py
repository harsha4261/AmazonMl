import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image
import os
import re
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# SMAPE metric
def calculate_smape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0 + 1e-8
    return np.mean(np.abs(predicted - actual) / denominator) * 100

def smape_lgb(y_pred, y_true):
    y_true = y_true.get_label()
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    smape = np.mean(np.abs(y_pred - y_true) / denominator)
    return 'smape', smape, False

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset for text and images
class ProductDataset(Dataset):
    def __init__(self, df, image_dir, tokenizer, is_train=True):
        self.df = df
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['catalog_content'])
        ipq_match = re.search(r'IPQ[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
        ipq = float(ipq_match.group(1)) if ipq_match else 1.0
        text = text.replace(f'IPQ: {ipq}', '').strip()

        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

        # Extract filename from image_link URL
        img_url = row['image_link']
        img_filename = os.path.basename(img_url)
        img_path = os.path.join(self.image_dir, img_filename)
        
        img = torch.zeros(3, 224, 224)  # Placeholder
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                img = image_transform(img)
            except:
                pass

        if self.is_train:
            price = np.log1p(row['price'])  # Log-transform price
            return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), img, ipq, price
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), img, ipq, row['sample_id']

# Feature extractor (BERT + ResNet)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased').to(device)
        self.resnet = models.resnet50(pretrained=True).to(device)
        self.resnet.fc = nn.Identity()

    def forward(self, input_ids, attention_mask, images):
        with torch.no_grad():
            text_feats = self.bert(input_ids, attention_mask).pooler_output  # [batch, 768]
            img_feats = self.resnet(images)  # [batch, 2048]
        return text_feats, img_feats

# Multimodal fusion model
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(768 + 2048 + 1, 512),  # Text + Image + IPQ
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, text_feats, img_feats, ipq):
        x = torch.cat([text_feats, img_feats, ipq.unsqueeze(1)], dim=1)
        return self.fc(x).squeeze()

# Ensemble model with boosting and neural network
class AdvancedEnsemble:
    def __init__(self):
        self.models = []
        self.weights = []
        self.model_smapes = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add_lgb_model(self):
        params = {
            'objective': 'regression', 'metric': 'None', 'num_leaves': 63, 'learning_rate': 0.03,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'max_depth': 8, 'min_data_in_leaf': 20, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1
        }
        return params

    def add_xgb_model(self):
        params = {
            'objective': 'reg:squarederror', 'eval_metric': 'mape', 'max_depth': 10,
            'learning_rate': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'min_child_weight': 3, 'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'tree_method': 'hist', 'verbosity': 0
        }
        return params

    def add_catboost_model(self):
        params = {
            'iterations': 1000, 'learning_rate': 0.03, 'depth': 8, 'l2_leaf_reg': 3,
            'min_data_in_leaf': 20, 'random_strength': 0.5, 'bagging_temperature': 0.2,
            'od_type': 'Iter', 'od_wait': 50, 'verbose': False
        }
        return params

    def train(self, train_df, image_dir, tokenizer, k_folds=5):
        print("\n" + "="*70 + "\nTRAINING ENSEMBLE MODELS\n" + "="*70)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_smapes = []

        extractor = FeatureExtractor().to(self.device)
        fusion_model = FusionModel().to(self.device)
        optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-3)

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            print(f"\nFold {fold+1}/{k_folds}")
            train_sub = train_df.iloc[train_idx]
            val_sub = train_df.iloc[val_idx]

            # Prepare dataset
            train_dataset = ProductDataset(train_sub, image_dir, tokenizer)
            val_dataset = ProductDataset(val_sub, image_dir, tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)

            # Extract features
            X_train, y_train = [], []
            X_val, y_val = [], []
            for data in train_loader:
                input_ids, attn_mask, imgs, ipq, price = [d.to(self.device) for d in data]
                text_feats, img_feats = extractor(input_ids, attn_mask, imgs)
                X_train.append(torch.cat([text_feats, img_feats, ipq.unsqueeze(1)], dim=1).cpu().numpy())
                y_train.append(price.cpu().numpy())
            for data in val_loader:
                input_ids, attn_mask, imgs, ipq, price = [d.to(self.device) for d in data]
                text_feats, img_feats = extractor(input_ids, attn_mask, imgs)
                X_val.append(torch.cat([text_feats, img_feats, ipq.unsqueeze(1)], dim=1).cpu().numpy())
                y_val.append(price.cpu().numpy())

            X_train = np.vstack(X_train)
            y_train = np.hstack(y_train)
            X_val = np.vstack(X_val)
            y_val = np.hstack(y_val)

            # Train fusion model
            fusion_model.train()
            for epoch in range(5):  # Reduced epochs for speed
                for data in train_loader:
                    input_ids, attn_mask, imgs, ipq, price = [d.to(self.device) for d in data]
                    text_feats, img_feats = extractor(input_ids, attn_mask, imgs)
                    pred = fusion_model(text_feats, img_feats, ipq)
                    loss = nn.MSELoss()(pred, price)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Train boosting models
            lgb_model = lgb.train(self.add_lgb_model(), lgb.Dataset(X_train, y_train),
                                  num_boost_round=500, valid_sets=[lgb.Dataset(X_val, y_val)],
                                  feval=smape_lgb, callbacks=[lgb.early_stopping(50)])
            xgb_model = xgb.train(self.add_xgb_model(), xgb.DMatrix(X_train, y_train),
                                  num_boost_round=500, evals=[(xgb.DMatrix(X_val, y_val), 'val')],
                                  early_stopping_rounds=50, verbose_eval=False)
            cat_model = CatBoostRegressor(**self.add_catboost_model()).fit(X_train, y_train,
                                                                          eval_set=(X_val, y_val),
                                                                          early_stopping_rounds=50, verbose=False)

            self.models = [('lgb', lgb_model), ('xgb', xgb_model), ('cat', cat_model), ('nn', fusion_model)]

            # Evaluate fold
            preds = []
            for name, model in self.models:
                if name == 'nn':
                    model.eval()
                    with torch.no_grad():
                        pred = []
                        for data in val_loader:
                            input_ids, attn_mask, imgs, ipq, _ = [d.to(self.device) for d in data]
                            text_feats, img_feats = extractor(input_ids, attn_mask, imgs)
                            pred.append(model(text_feats, img_feats, ipq).cpu().numpy())
                        pred = np.hstack(pred)
                else:
                    pred = model.predict(X_val) if name != 'xgb' else model.predict(xgb.DMatrix(X_val))
                preds.append(pred)
            ensemble_pred = np.average(preds, axis=0, weights=[0.25, 0.25, 0.25, 0.25])
            smape = calculate_smape(np.expm1(y_val), np.expm1(ensemble_pred))
            fold_smapes.append(smape)
            print(f"Fold {fold+1} SMAPE: {smape:.2f}%")

        self.model_smapes['ensemble'] = np.mean(fold_smapes)
        print(f"\nMean CV SMAPE: {self.model_smapes['ensemble']:.2f}%")

    def predict(self, test_df, image_dir, tokenizer):
        dataset = ProductDataset(test_df, image_dir, tokenizer, is_train=False)
        loader = DataLoader(dataset, batch_size=32)
        extractor = FeatureExtractor().to(self.device)

        X_test, sample_ids = [], []
        for data in loader:
            input_ids, attn_mask, imgs, ipq, sid = [d.to(self.device) if isinstance(d, torch.Tensor) else sid for d in data]
            text_feats, img_feats = extractor(input_ids, attn_mask, imgs)
            X_test.append(torch.cat([text_feats, img_feats, ipq.unsqueeze(1)], dim=1).cpu().numpy())
            sample_ids.extend(sid)

        X_test = np.vstack(X_test)

        preds = []
        for name, model in self.models:
            if name == 'nn':
                model.eval()
                with torch.no_grad():
                    pred = []
                    for data in loader:
                        input_ids, attn_mask, imgs, ipq, _ = [d.to(self.device) if isinstance(d, torch.Tensor) else _ for d in data]
                        text_feats, img_feats = extractor(input_ids, attn_mask, imgs)
                        pred.append(model(text_feats, img_feats, ipq).cpu().numpy())
                    pred = np.hstack(pred)
            else:
                pred = model.predict(X_test) if name != 'xgb' else model.predict(xgb.DMatrix(X_test))
            preds.append(pred)
        final_preds = np.expm1(np.average(preds, axis=0, weights=[0.25, 0.25, 0.25, 0.25]))
        final_preds = np.maximum(final_preds, 0.01)
        return pd.DataFrame({'sample_id': sample_ids, 'price': final_preds})

# Main execution
def main():
    print("="*70 + "\nADVANCED MULTIMODAL ENSEMBLE\n" + "="*70)
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    image_dir = './images/'
    os.makedirs(image_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Download images (assuming src/utils.py provides download_images)
    from src.utils import download_images
    download_images(train_df['image_link'].tolist() + test_df['image_link'].tolist(), image_dir)

    ensemble = AdvancedEnsemble()
    ensemble.train(train_df, image_dir, tokenizer)
    
    submission = ensemble.predict(test_df, image_dir, tokenizer)
    submission.to_csv('submission.csv', index=False)
    
    print("\n" + "="*70 + "\nFINAL RESULTS\n" + "="*70)
    print(f"✓ Submission file created: submission.csv")
    print(f"✓ Number of predictions: {len(submission):,}")
    print(f"✓ Predicted price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
    print(f"✓ Mean predicted price: ${submission['price'].mean():.2f}")
    print(f"✓ Mean CV SMAPE: {ensemble.model_smapes['ensemble']:.2f}%")
    rating = "🏆 EXCELLENT" if ensemble.model_smapes['ensemble'] < 15 else "✅ GOOD" if ensemble.model_smapes['ensemble'] < 20 else "👍 ACCEPTABLE" if ensemble.model_smapes['ensemble'] < 30 else "⚠️ NEEDS IMPROVEMENT"
    print(f"Rating: {rating}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()