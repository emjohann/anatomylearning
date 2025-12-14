# -*- coding: utf-8 -*-
"""
@author: Prithila Angkan
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import warnings
import os
warnings.filterwarnings('ignore')

current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

select_label = 'CL_Total' # 'CL_All_Average', 'ICL_Average', 'ECL_Average'
threshold = 22  # Set the threshold accordingly

# Define save folder
save_folder = f'./LOSO/LOSO_Results_{select_label}_TH_{threshold}/{current_datetime}/'
os.makedirs(save_folder, exist_ok=True)
print(f"\nFiles will be saved to: {save_folder}\n")

# Load and preprocess data
df = pd.read_csv('./Combined_Data2.csv')
df2 = df.drop(df.columns[[1,2,3,4,5,6,7,88,89,90,91,92,93,94,95,99]], axis=1)

# Drop rows with NaN
df2 = df2.dropna().reset_index(drop=True)

# Round and binarize
cols_to_round = ['CL_All_Average', 'ICL_Average', 'ECL_Average']
df2[cols_to_round] = np.round(df2[cols_to_round]).astype(int)

value_counts = df2[select_label].value_counts()
value_sums = (value_counts.index.to_numpy() * value_counts.values).sum()
print(f"\n{select_label} distribution before binarization:")
print(value_counts)
print("Sum of each unique value × count:", value_sums)

# Binarize based on threshold
df2['CL_All_Average'] = df2['CL_All_Average'].apply(lambda x: 0 if x <= threshold else 1) # 4
df2['ICL_Average'] = df2['ICL_Average'].apply(lambda x: 0 if x <= threshold else 1) # 5
df2['ECL_Average'] = df2['ECL_Average'].apply(lambda x: 0 if x <= threshold else 1) # 4
df2['CL_Total'] = df2['CL_Total'].apply(lambda x: 0 if x <= threshold else 1) # cl_total all threshold 35 | only icl and ecl thrheshold 22

# Get participants
participant_ids = df2['Participant ID'].unique()
print(f"\nNumber of participants: {len(participant_ids)}")

# Feature columns
feature_cols = [col for col in df2.columns if 'Tx' in col]
print(f"Number of features: {len(feature_cols)}")

# Choose target variable
target = select_label
X = df2[feature_cols].values
y = df2[target].values

# Z-score normalization for each row/sample
print("\n" + "="*80)
print("Applying z-score (row-wise)")
print("="*80)

print(f"Original data - Mean: {X.mean():.4f}, Std: {X.std():.4f}")

# Apply z-score normalization to each row
X_normalized = np.zeros_like(X)
for i in range(X.shape[0]):
    row_mean = X[i].mean()
    row_std = X[i].std()
    if row_std > 0:
        X_normalized[i] = (X[i] - row_mean) / row_std
    else:
        X_normalized[i] = X[i] - row_mean
    
# Handle any remaining NaN values
X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=0.0, neginf=0.0)

X = X_normalized
print("Z-score normalization applied to each sample (row)")
print(f"Normalized data - Mean: {X.mean():.4f}, Std: {X.std():.4f}")
print(f"Shape: {X.shape}")

# Class distribution analysis
print("\n" + "="*80)
print("Class distribution analysis")
print("="*80)

# Overall
print(f"\nOverall: Total={len(y)}, Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}")
print(f"  Class balance: {np.sum(y==1)/len(y)*100:.1f}% Class 1")

# Per participant
print(f"\nPer-participant distribution:")
only_class0 = []
only_class1 = []
both_classes = []

for pid in participant_ids:
    mask = df2['Participant ID'] == pid
    y_part = y[mask]
    class_0 = np.sum(y_part == 0)
    class_1 = np.sum(y_part == 1)
    
    if class_1 == 0:
        only_class0.append(pid)
        status = "Only Class 0"
    elif class_0 == 0:
        only_class1.append(pid)
        status = "Only Class 1"
    else:
        both_classes.append(pid)
        status = "Both classes"
    
    print(f"  P{pid}: Total={len(y_part)}, C0={class_0}, C1={class_1} - {status}")

print(f"\nSummary:")
print(f"Participants with both classes: {len(both_classes)}")
print(f" Participants with only Class 0: {len(only_class0)}")
print(f"Participants with only Class 1: {len(only_class1)}")

if only_class0:
    print(f"\n  Participants with only Class 0: {only_class0}")
if only_class1:
    print(f"  Participants with only Class 1: {only_class1}")

print(f"\nTarget: {target}")
print(f"Class 0: {np.sum(y==0)} samples, Class 1: {np.sum(y==1)} samples")

# Initialize models
models = {
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

# Fully Connected Neural Network (MLP/FCNN)
class FCNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(FCNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Convolutional Neural Network (CNN)
class CNNClassifier(nn.Module):
    def __init__(self, input_dim):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        conv_output_size = 64 * (input_dim // 4)
        
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

def train_deep_learning(X_train, y_train, X_test, y_test, model_type='FCNN'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    if model_type == 'CNN':
        model = CNNClassifier(input_dim=X_train.shape[1]).to(device)
    else:
        model = FCNNClassifier(input_dim=X_train.shape[1], hidden_dim=128).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    batch_size = 32
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_epoch = 0
    best_f1_macro = 0
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    
    print(f"{model_type} Training:", end=" ")
    
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            predictions = (outputs >= 0.5).float()
            
            y_pred = predictions.cpu().numpy().flatten()
            y_true = y_test_tensor.cpu().numpy().flatten()
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            if (epoch + 1) % 10 == 0:
                print(f"E{epoch+1}:F1={f1_macro:.3f}", end=" ")
            
            if f1_macro > best_f1_macro:
                best_epoch = epoch + 1
                best_f1_macro = f1_macro
                best_accuracy = acc
                best_precision = prec
                best_recall = rec
    
    print(f"→ Best@E{best_epoch}")
    
    return best_epoch, best_accuracy, best_precision, best_recall, best_f1_macro

# Store results
ml_results = []
fcnn_results = []
cnn_results = []

print("\n" + "="*80)
print("LOSO started")
print("="*80)

# LOSO
for fold, test_participant in enumerate(participant_ids, 1):
    print(f"\nFold {fold}/{len(participant_ids)}: Testing on Participant {test_participant}")
    
    # Split data based on participant
    train_mask = df2['Participant ID'] != test_participant
    test_mask = df2['Participant ID'] == test_participant
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Check class distribution
    train_class_0 = np.sum(y_train == 0)
    train_class_1 = np.sum(y_train == 1)
    test_class_0 = np.sum(y_test == 0)
    test_class_1 = np.sum(y_test == 1)
    
    print(f"  Train: {len(X_train)} samples (Class 0: {train_class_0}, Class 1: {train_class_1})")
    print(f"  Test:  {len(X_test)} samples (Class 0: {test_class_0}, Class 1: {test_class_1})")
    
    X_train_scaled = X_train
    X_test_scaled = X_test
    
    # Test each ML model
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        ml_results.append({
            'Model': model_name,
            'Test_Participant': test_participant,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Macro': f1_macro
        })
        
        print(f"  {model_name:20s}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1-Macro={f1_macro:.4f}")
    
    # Test FCNN/MLP model
    best_epoch, fcnn_acc, fcnn_prec, fcnn_rec, fcnn_f1_macro = train_deep_learning(
        X_train_scaled, y_train, X_test_scaled, y_test, model_type='FCNN'
    )
    
    fcnn_results.append({
        'Test_Participant': test_participant,
        'Best_Epoch': best_epoch,
        'Best_Accuracy': fcnn_acc,
        'Best_Precision': fcnn_prec,
        'Best_Recall': fcnn_rec,
        'Best_F1_Macro': fcnn_f1_macro
    })
    
    print(f"  {'FCNN':20s}: Epoch={best_epoch}, Acc={fcnn_acc:.4f}, Prec={fcnn_prec:.4f}, Rec={fcnn_rec:.4f}, F1-Macro={fcnn_f1_macro:.4f}")
    
    # Test CNN model
    best_epoch, cnn_acc, cnn_prec, cnn_rec, cnn_f1_macro = train_deep_learning(
        X_train_scaled, y_train, X_test_scaled, y_test, model_type='CNN'
    )
    
    cnn_results.append({
        'Test_Participant': test_participant,
        'Best_Epoch': best_epoch,
        'Best_Accuracy': cnn_acc,
        'Best_Precision': cnn_prec,
        'Best_Recall': cnn_rec,
        'Best_F1_Macro': cnn_f1_macro
    })
    
    print(f"  {'CNN':20s}: Epoch={best_epoch}, Acc={cnn_acc:.4f}, Prec={cnn_prec:.4f}, Rec={cnn_rec:.4f}, F1-Macro={cnn_f1_macro:.4f}")

# Check if we have valid results
if len(ml_results) == 0 or len(fcnn_results) == 0 or len(cnn_results) == 0:
    print("\n" + "="*80)
    print("ERROR: No valid folds to evaluate!")
    print("="*80)
    exit()

# Save results
print("\n" + "="*80)
print("Saving results")
print("="*80)

# Save FCNN results
fcnn_df = pd.DataFrame(fcnn_results)
fcnn_mean = fcnn_df[['Best_Accuracy', 'Best_Precision', 'Best_Recall', 'Best_F1_Macro']].mean()
fcnn_std = fcnn_df[['Best_Accuracy', 'Best_Precision', 'Best_Recall', 'Best_F1_Macro']].std()

mean_row = pd.DataFrame({
    'Test_Participant': ['Mean'], 'Best_Epoch': ['-'],
    'Best_Accuracy': [fcnn_mean['Best_Accuracy']], 'Best_Precision': [fcnn_mean['Best_Precision']],
    'Best_Recall': [fcnn_mean['Best_Recall']], 'Best_F1_Macro': [fcnn_mean['Best_F1_Macro']]
})
std_row = pd.DataFrame({
    'Test_Participant': ['Std'], 'Best_Epoch': ['-'],
    'Best_Accuracy': [fcnn_std['Best_Accuracy']], 'Best_Precision': [fcnn_std['Best_Precision']],
    'Best_Recall': [fcnn_std['Best_Recall']], 'Best_F1_Macro': [fcnn_std['Best_F1_Macro']]
})
fcnn_df = pd.concat([fcnn_df, mean_row, std_row], ignore_index=True)
fcnn_df.to_csv(os.path.join(save_folder, f'FCNN_{target}_results.csv'), index=False, encoding='utf-8-sig')
print(f"Saved: FCNN_{target}_results.csv")

# Save CNN results
cnn_df = pd.DataFrame(cnn_results)
cnn_mean = cnn_df[['Best_Accuracy', 'Best_Precision', 'Best_Recall', 'Best_F1_Macro']].mean()
cnn_std = cnn_df[['Best_Accuracy', 'Best_Precision', 'Best_Recall', 'Best_F1_Macro']].std()

mean_row = pd.DataFrame({
    'Test_Participant': ['Mean'], 'Best_Epoch': ['-'],
    'Best_Accuracy': [cnn_mean['Best_Accuracy']], 'Best_Precision': [cnn_mean['Best_Precision']],
    'Best_Recall': [cnn_mean['Best_Recall']], 'Best_F1_Macro': [cnn_mean['Best_F1_Macro']]
})
std_row = pd.DataFrame({
    'Test_Participant': ['Std'], 'Best_Epoch': ['-'],
    'Best_Accuracy': [cnn_std['Best_Accuracy']], 'Best_Precision': [cnn_std['Best_Precision']],
    'Best_Recall': [cnn_std['Best_Recall']], 'Best_F1_Macro': [cnn_std['Best_F1_Macro']]
})
cnn_df = pd.concat([cnn_df, mean_row, std_row], ignore_index=True)
cnn_df.to_csv(os.path.join(save_folder, f'CNN_{target}_results.csv'), index=False, encoding='utf-8-sig')
print(f"Saved: CNN_{target}_results.csv")

# Save Machine Learning results
ml_df = pd.DataFrame(ml_results)

for model_name in models.keys():
    model_df = ml_df[ml_df['Model'] == model_name].copy()
    model_df = model_df.drop('Model', axis=1)
    
    model_mean = model_df[['Accuracy', 'Precision', 'Recall', 'F1_Macro']].mean()
    model_std = model_df[['Accuracy', 'Precision', 'Recall', 'F1_Macro']].std()
    
    mean_row = pd.DataFrame({
        'Test_Participant': ['Mean'],
        'Accuracy': [model_mean['Accuracy']], 'Precision': [model_mean['Precision']],
        'Recall': [model_mean['Recall']], 'F1_Macro': [model_mean['F1_Macro']]
    })
    std_row = pd.DataFrame({
        'Test_Participant': ['Std'],
        'Accuracy': [model_std['Accuracy']], 'Precision': [model_std['Precision']],
        'Recall': [model_std['Recall']], 'F1_Macro': [model_std['F1_Macro']]
    })
    
    model_df = pd.concat([model_df, mean_row, std_row], ignore_index=True)
    model_filename = os.path.join(save_folder, f'{model_name.replace(" ", "_")}_{target}_results.csv')
    model_df.to_csv(model_filename, index=False, encoding='utf-8-sig')
    print(f"✓ Saved: {model_filename}")

# Create summary table
print("\n" + "="*80)
print("Final Summary")
print("="*80)

summary_data = []

# FCNN summary
summary_data.append({
    'Model': 'FCNN',
    'Accuracy': f"{fcnn_mean['Best_Accuracy']:.4f} ± {fcnn_std['Best_Accuracy']:.4f}",
    'Precision': f"{fcnn_mean['Best_Precision']:.4f} ± {fcnn_std['Best_Precision']:.4f}",
    'Recall': f"{fcnn_mean['Best_Recall']:.4f} ± {fcnn_std['Best_Recall']:.4f}",
    'F1_Macro': f"{fcnn_mean['Best_F1_Macro']:.4f} ± {fcnn_std['Best_F1_Macro']:.4f}"
})

# CNN summary
summary_data.append({
    'Model': 'CNN',
    'Accuracy': f"{cnn_mean['Best_Accuracy']:.4f} ± {cnn_std['Best_Accuracy']:.4f}",
    'Precision': f"{cnn_mean['Best_Precision']:.4f} ± {cnn_std['Best_Precision']:.4f}",
    'Recall': f"{cnn_mean['Best_Recall']:.4f} ± {cnn_std['Best_Recall']:.4f}",
    'F1_Macro': f"{cnn_mean['Best_F1_Macro']:.4f} ± {cnn_std['Best_F1_Macro']:.4f}"
})

# ML models summary
for model_name in models.keys():
    model_data = ml_df[ml_df['Model'] == model_name]
    mean_acc = model_data['Accuracy'].mean()
    std_acc = model_data['Accuracy'].std()
    mean_prec = model_data['Precision'].mean()
    std_prec = model_data['Precision'].std()
    mean_rec = model_data['Recall'].mean()
    std_rec = model_data['Recall'].std()
    mean_f1 = model_data['F1_Macro'].mean()
    std_f1 = model_data['F1_Macro'].std()
    
    summary_data.append({
        'Model': model_name,
        'Accuracy': f"{mean_acc:.4f} ± {std_acc:.4f}",
        'Precision': f"{mean_prec:.4f} ± {std_prec:.4f}",
        'Recall': f"{mean_rec:.4f} ± {std_rec:.4f}",
        'F1_Macro': f"{mean_f1:.4f} ± {std_f1:.4f}"
    })

summary_df = pd.DataFrame(summary_data)
summary_filename = os.path.join(save_folder, f'Summary_{target}.csv')
summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
print(f"\nSaved summary: {summary_filename}")

print("\n" + summary_df.to_string(index=False))

