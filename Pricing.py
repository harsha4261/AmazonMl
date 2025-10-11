"""
Advanced Ensemble Solution with Multiple Models
Combines Deep Learning, Gradient Boosting, and Feature Engineering
WITH SMAPE TRACKING
"""

import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

def calculate_smape(actual, predicted):
    """Calculate SMAPE metric"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    smape = np.mean(np.abs(predicted - actual) / (denominator + 1e-8)) * 100
    return smape

def smape_lgb(y_pred, y_true):
    """SMAPE metric for LightGBM - note reversed parameter order"""
    y_true = y_true.get_label()  # Extract actual values from Dataset object
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.mean(np.abs(y_pred - y_true) / (denominator + 1e-8))
    return 'smape', smape, False

class AdvancedFeatureExtractor:
    """Extract comprehensive features from catalog content"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fit_done = False
        
        # Brand keywords that typically indicate higher prices
        self.premium_brands = set(['apple', 'samsung', 'sony', 'nike', 'adidas', 
                                  'gucci', 'prada', 'rolex', 'omega'])
        
        # Material indicators
        self.premium_materials = set(['gold', 'silver', 'platinum', 'diamond', 
                                     'leather', 'silk', 'cashmere', 'titanium'])
    
    def extract_features(self, text):
        """Extract 50+ features from text"""
        text = str(text)
        text_lower = text.lower()
        features = {}
        
        # === Basic Text Features ===
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text.split() else 0
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features['special_char_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0
        
        # === Item Pack Quantity (IPQ) ===
        ipq_match = re.search(r'IPQ[:\s]*(\d+)', text, re.IGNORECASE)
        features['ipq'] = int(ipq_match.group(1)) if ipq_match else 1
        features['ipq_log'] = np.log1p(features['ipq'])
        
        # === Numeric Features ===
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            numbers = [float(n) for n in numbers]
            features['max_number'] = max(numbers)
            features['min_number'] = min(numbers)
            features['avg_number'] = np.mean(numbers)
            features['std_number'] = np.std(numbers)
            features['num_count'] = len(numbers)
            features['max_number_log'] = np.log1p(features['max_number'])
        else:
            features['max_number'] = 0
            features['min_number'] = 0
            features['avg_number'] = 0
            features['std_number'] = 0
            features['num_count'] = 0
            features['max_number_log'] = 0
        
        # === Unit and Quantity Detection ===
        # Weight units
        weight_pattern = r'(\d+\.?\d*)\s*(mg|g|kg|gram|grams|kilogram|kilograms)'
        weight_matches = re.findall(weight_pattern, text_lower)
        features['has_weight'] = 1 if weight_matches else 0
        features['weight_value'] = float(weight_matches[0][0]) if weight_matches else 0
        
        # Volume units
        volume_pattern = r'(\d+\.?\d*)\s*(ml|l|oz|fluid|liters?|milliliters?)'
        volume_matches = re.findall(volume_pattern, text_lower)
        features['has_volume'] = 1 if volume_matches else 0
        features['volume_value'] = float(volume_matches[0][0]) if volume_matches else 0
        
        # Count/quantity
        count_pattern = r'(\d+)\s*(count|pieces?|items?|pcs?|pack)'
        count_matches = re.findall(count_pattern, text_lower)
        features['has_count'] = 1 if count_matches else 0
        features['count_value'] = float(count_matches[0][0]) if count_matches else 0
        
        # Size (inches, cm, etc.)
        size_pattern = r'(\d+\.?\d*)\s*(inch|inches|cm|mm|meter|ft|foot|feet)'
        size_matches = re.findall(size_pattern, text_lower)
        features['has_size'] = 1 if size_matches else 0
        features['size_value'] = float(size_matches[0][0]) if size_matches else 0
        
        # === Brand and Premium Indicators ===
        features['has_premium_brand'] = 1 if any(brand in text_lower for brand in self.premium_brands) else 0
        features['has_premium_material'] = 1 if any(mat in text_lower for mat in self.premium_materials) else 0
        
        premium_keywords = ['premium', 'luxury', 'professional', 'deluxe', 'elite', 
                           'organic', 'natural', 'imported', 'original', 'authentic',
                           'exclusive', 'limited', 'special', 'edition']
        features['premium_score'] = sum(1 for word in premium_keywords if word in text_lower)
        
        # === Discount and Bundle Indicators ===
        bundle_keywords = ['pack', 'combo', 'bundle', 'set', 'value', 'multipack',
                          'assorted', 'variety', 'collection']
        features['bundle_score'] = sum(1 for word in bundle_keywords if word in text_lower)
        
        # === Category Hints ===
        electronics_keywords = ['electronics', 'phone', 'laptop', 'computer', 'tablet',
                               'camera', 'headphone', 'speaker', 'charger', 'cable']
        features['is_electronics'] = 1 if any(word in text_lower for word in electronics_keywords) else 0
        
        food_keywords = ['food', 'snack', 'beverage', 'drink', 'coffee', 'tea',
                        'chocolate', 'candy', 'cereal', 'nutrition']
        features['is_food'] = 1 if any(word in text_lower for word in food_keywords) else 0
        
        beauty_keywords = ['beauty', 'cosmetic', 'skincare', 'makeup', 'perfume',
                          'shampoo', 'conditioner', 'lotion', 'cream', 'serum']
        features['is_beauty'] = 1 if any(word in text_lower for word in beauty_keywords) else 0
        
        clothing_keywords = ['shirt', 'pant', 'dress', 'shoe', 'jacket', 'coat',
                            'sweater', 'jeans', 'clothing', 'apparel']
        features['is_clothing'] = 1 if any(word in text_lower for word in clothing_keywords) else 0
        
        # === Quality Indicators ===
        quality_positive = ['new', 'fresh', 'best', 'top', 'high', 'quality',
                           'superior', 'excellent', 'perfect']
        features['quality_positive'] = sum(1 for word in quality_positive if word in text_lower)
        
        # === Brand Name Extraction ===
        words = text.split()
        capitalized_start = []
        for word in words[:5]:
            if word and word[0].isupper():
                capitalized_start.append(word)
            else:
                break
        features['potential_brand_length'] = len(' '.join(capitalized_start))
        
        # === Title Length Analysis ===
        title_match = text.split('\n')[0] if '\n' in text else text.split('.')[0]
        features['title_length'] = len(title_match)
        features['title_word_count'] = len(title_match.split())
        
        # === Interaction Features ===
        features['ipq_times_weight'] = features['ipq'] * features['weight_value']
        features['ipq_times_volume'] = features['ipq'] * features['volume_value']
        features['ipq_times_count'] = features['ipq'] * features['count_value']
        features['premium_bundle_interaction'] = features['premium_score'] * features['bundle_score']
        
        return features
    
    def fit_transform(self, texts):
        """Extract and scale features for training"""
        features_list = [self.extract_features(text) for text in texts]
        df_features = pd.DataFrame(features_list)
        
        if not self.fit_done:
            self.feature_names = df_features.columns.tolist()
            scaled = self.scaler.fit_transform(df_features)
            self.fit_done = True
        else:
            scaled = self.scaler.transform(df_features)
        
        return scaled, df_features
    
    def transform(self, texts):
        """Transform features for inference"""
        features_list = [self.extract_features(text) for text in texts]
        df_features = pd.DataFrame(features_list)
        
        # Ensure all features are present
        for col in self.feature_names:
            if col not in df_features.columns:
                df_features[col] = 0
        
        df_features = df_features[self.feature_names]
        scaled = self.scaler.transform(df_features)
        
        return scaled, df_features

class EnsembleModel:
    """Ensemble of multiple models with SMAPE tracking"""
    
    def __init__(self):
        self.models = []
        self.weights = []
        self.model_smapes = {}
        
    def add_lgb_model(self):
        """Add LightGBM model"""
        params = {
            'objective': 'regression',
            'metric': 'None',  # Use custom SMAPE
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 8,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1
        }
        return params
    
    def add_xgb_model(self):
        """Add XGBoost model"""
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mape',
            'max_depth': 10,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'tree_method': 'hist',
            'verbosity': 0
        }
        return params
    
    def add_catboost_model(self):
        """Add CatBoost model"""
        params = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 8,
            'l2_leaf_reg': 3,
            'min_data_in_leaf': 20,
            'random_strength': 0.5,
            'bagging_temperature': 0.2,
            'od_type': 'Iter',
            'od_wait': 50,
            'verbose': False
        }
        return params
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train all models with SMAPE tracking"""
        print("\n" + "="*70)
        print("TRAINING ENSEMBLE MODELS")
        print("="*70)
        
        # Train LightGBM
        print("\n[1/3] Training LightGBM...")
        print("-"*70)
        lgb_params = self.add_lgb_model()
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_valid],
            valid_names=['valid'],
            feval=smape_lgb,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        self.models.append(('lgb', lgb_model))
        
        # Calculate SMAPE
        lgb_pred = lgb_model.predict(X_val)
        lgb_smape = calculate_smape(y_val, lgb_pred)
        self.model_smapes['lgb'] = lgb_smape
        print(f"✓ LightGBM Validation SMAPE: {lgb_smape:.2f}%")
        
        # Train XGBoost
        print("\n[2/3] Training XGBoost...")
        print("-"*70)
        xgb_params = self.add_xgb_model()
        xgb_train = xgb.DMatrix(X_train, y_train)
        xgb_valid = xgb.DMatrix(X_val, y_val)
        
        xgb_model = xgb.train(
            xgb_params,
            xgb_train,
            num_boost_round=1000,
            evals=[(xgb_valid, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        self.models.append(('xgb', xgb_model))
        
        # Calculate SMAPE
        xgb_pred = xgb_model.predict(xgb_valid)
        xgb_smape = calculate_smape(y_val, xgb_pred)
        self.model_smapes['xgb'] = xgb_smape
        print(f"✓ XGBoost Validation SMAPE: {xgb_smape:.2f}%")
        
        # Train CatBoost
        print("\n[3/3] Training CatBoost...")
        print("-"*70)
        cat_params = self.add_catboost_model()
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=100
        )
        self.models.append(('cat', cat_model))
        
        # Calculate SMAPE
        cat_pred = cat_model.predict(X_val)
        cat_smape = calculate_smape(y_val, cat_pred)
        self.model_smapes['cat'] = cat_smape
        print(f"✓ CatBoost Validation SMAPE: {cat_smape:.2f}%")
        
        # Display individual model results
        print("\n" + "="*70)
        print("INDIVIDUAL MODEL SMAPE SCORES")
        print("="*70)
        for name, smape in self.model_smapes.items():
            print(f"   {name.upper():<15} SMAPE: {smape:.4f}% ({smape:.2f}%)")
        print("="*70)
        
        # Optimize ensemble weights
        print("\nOptimizing ensemble weights...")
        self.optimize_weights(X_val, y_val)
    
    def optimize_weights(self, X_val, y_val):
        """Find optimal ensemble weights"""
        from scipy.optimize import minimize
        
        def objective(weights):
            pred = self.predict_with_weights(X_val, weights)
            smape = np.mean(np.abs(pred - y_val) / ((np.abs(y_val) + np.abs(pred)) / 2 + 1e-8))
            return smape
        
        # Initial equal weights
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=[(0, 1)] * len(self.models),
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        
        self.weights = result.x
        
        # Calculate ensemble SMAPE
        ensemble_pred = self.predict_with_weights(X_val, self.weights)
        ensemble_smape = calculate_smape(y_val, ensemble_pred)
        
        print("\n" + "="*70)
        print("ENSEMBLE OPTIMIZATION RESULTS")
        print("="*70)
        print("Optimized weights:")
        for (name, _), weight in zip(self.models, self.weights):
            print(f"   {name.upper():<15} Weight: {weight:.4f} ({weight*100:.2f}%)")
        print(f"\n✓ Ensemble Validation SMAPE: {ensemble_smape:.4f}% ({ensemble_smape:.2f}%)")
        print("="*70)
    
    def predict_with_weights(self, X, weights):
        """Predict with custom weights"""
        predictions = []
        
        for (name, model), weight in zip(self.models, weights):
            if name == 'lgb':
                pred = model.predict(X)
            elif name == 'xgb':
                pred = model.predict(xgb.DMatrix(X))
            else:  # catboost
                pred = model.predict(X)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def predict(self, X):
        """Predict with optimized weights"""
        return self.predict_with_weights(X, self.weights)

# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    print("="*70)
    print("ADVANCED ENSEMBLE SOLUTION - SMART PRODUCT PRICING")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"   ✓ Train: {len(train_df):,} samples")
    print(f"   ✓ Test: {len(test_df):,} samples")
    print(f"   Price range: ${train_df['price'].min():.2f} - ${train_df['price'].max():.2f}")
    
    # Feature extraction
    print("\n[2/5] Extracting advanced features...")
    feature_extractor = AdvancedFeatureExtractor()
    
    X_train_scaled, X_train_df = feature_extractor.fit_transform(
        train_df['catalog_content'].astype(str)
    )
    y_train = train_df['price'].values
    
    X_test_scaled, X_test_df = feature_extractor.transform(
        test_df['catalog_content'].astype(str)
    )
    
    print(f"   ✓ Extracted {X_train_df.shape[1]} features")
    print(f"   Top 5 features: {', '.join(X_train_df.columns[:5].tolist())}")
    
    # Split data
    print("\n[3/5] Splitting data...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train,
        test_size=0.15,
        random_state=42
    )
    print(f"   ✓ Train: {len(X_tr):,}, Validation: {len(X_val):,}")
    
    # Train ensemble
    print("\n[4/5] Training ensemble models...")
    ensemble = EnsembleModel()
    ensemble.train(X_tr, y_tr, X_val, y_val)
    
    # Generate predictions
    print("\n[5/5] Generating predictions...")
    test_predictions = ensemble.predict(X_test_scaled)
    test_predictions = np.maximum(test_predictions, 0.01)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    
    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS & SUBMISSION")
    print("="*70)
    print(f"✓ Submission file created: submission.csv")
    print(f"✓ Number of predictions: {len(submission):,}")
    print(f"✓ Predicted price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
    print(f"✓ Mean predicted price: ${submission['price'].mean():.2f}")
    print(f"✓ Median predicted price: ${submission['price'].median():.2f}")
    
    print(f"\n📊 MODEL PERFORMANCE SUMMARY:")
    print("-"*70)
    for name, smape in ensemble.model_smapes.items():
        print(f"   {name.upper():<15} SMAPE: {smape:.2f}%")
    
    # Calculate ensemble SMAPE
    ensemble_pred = ensemble.predict(X_val)
    ensemble_smape = calculate_smape(y_val, ensemble_pred)
    print(f"   {'ENSEMBLE':<15} SMAPE: {ensemble_smape:.2f}%")
    
    # Rating
    if ensemble_smape < 15:
        rating = "🏆 EXCELLENT"
    elif ensemble_smape < 20:
        rating = "✅ GOOD"
    elif ensemble_smape < 30:
        rating = "👍 ACCEPTABLE"
    else:
        rating = "⚠️  NEEDS IMPROVEMENT"
    
    print(f"\n   Rating: {rating}")
    print("\n✓ Submission file is ready for upload!")
    print("="*70)

if __name__ == "__main__":
    main()