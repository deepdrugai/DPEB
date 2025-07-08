#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 02:45:20 2025
@author: saiful
"""
import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import ast
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
embed_type = 'ESM'
import sys
sys.stdout = open(f"console_outputs/console_output_{embed_type}2.txt", "w")
# sys.stdout = open("console_outputs/console_output", "w")

print(f"Embedding type: {embed_type}")

#%% Load and Preprocess Data

def load_and_filter_data(protein_file, embedding_file):
    try:
        # Load protein families
        df_protein_families = pd.read_csv(protein_file)
        df_selected = df_protein_families[['Protein_ID', 'Family']]
        df_filtered = df_selected[~df_selected['Family'].isin(["No SIMILARITY comment found", "No family label found"])]
        
        # Filter families with >60 occurrences
        family_frequencies = df_filtered['Family'].value_counts()
        frequent_families = family_frequencies[family_frequencies > 60].index
        df_filtered = df_filtered[df_filtered['Family'].isin(frequent_families)]
        
        # Load AlphaFold embeddings
        df_esm = pd.read_csv(embedding_file)
        df_esm['ESM_Embeddings'] = df_esm['ESM_Embeddings'].apply(ast.literal_eval)
        
        # Merge dataframes
        merged_df = pd.merge(df_esm, df_filtered, left_on='ProteinID', right_on='Protein_ID', how='inner')
        
        print(f"Merged DataFrame shape: {merged_df.shape}")
        print(f"Unique families: {df_filtered['Family'].nunique()}")
        return merged_df
    except Exception as e:
        print(f"Error in load_and_filter_data: {e}")
        raise


# Load data
protein_file = "/home/saiful/ePPI_dgl/clustering/protein_families23k.csv"
# embedding_file = "/data/saiful/ePPI/alphafold_eppi_embeddings/eppi_alphafold_aggregated_embeddings.csv"
embedding_file = "/data/saiful/ePPI/ProteinID_proteinSEQ_ESM_emb.csv"


try:
    merged_df = load_and_filter_data(protein_file, embedding_file)
except Exception as e:
    print(f"Failed to load data: {e}")
    exit()

embeddings_raw = np.array(merged_df['ESM_Embeddings'].tolist())
true_labels = merged_df['Family']
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)
print(f"Raw {embed_type} Embeddings shape: {embeddings_raw.shape}")
print(f"Number of unique labels: {len(np.unique(true_labels_encoded))}")

#%% Plot Raw Embeddings
def plot_embeddings(embeddings, labels, title, save_path, label_type="Class"):
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='tab20', s=20, alpha=0.8)
        plt.colorbar(scatter, label=label_type)
        plt.title(title)
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Error in plot_embeddings: {e}")
        raise

# Plot raw embeddings with true labels
plot_embeddings(embeddings_raw, true_labels_encoded, f"t-SNE Visualization of Raw {embed_type} Embeddings", 
                f"/home/saiful/ePPI_dgl/clustering/figures_paper/raw_embeddings_tsne_{embed_type}.png")

#%% K-means Clustering on Raw Embeddings
def evaluate_kmeans(embeddings, true_labels_encoded, label_encoder, title, save_path):
    try:
        from sklearn.decomposition import PCA

        # Apply PCA to retain 95% of the variance
        pca = PCA(n_components=0.95, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)
        print(f"PCA reduced shape for {title}: {embeddings_reduced.shape}")
        
        # Perform K-means clustering on PCA-reduced embeddings
        k = len(np.unique(true_labels_encoded))
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_reduced)
        
        # Map cluster labels to true labels via majority voting
        cluster_to_label = {}
        for cluster in np.unique(cluster_labels):
            indices = np.where(cluster_labels == cluster)[0]
            true_labels_cluster = true_labels_encoded[indices]
            majority_label = np.bincount(true_labels_cluster).argmax()
            cluster_to_label[cluster] = majority_label
        
        mapped_labels = np.array([cluster_to_label[cluster] for cluster in cluster_labels])
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(true_labels_encoded, mapped_labels),
            'precision': precision_score(true_labels_encoded, mapped_labels, average='weighted', zero_division=0),
            'recall': recall_score(true_labels_encoded, mapped_labels, average='weighted', zero_division=0),
            'f1_score': f1_score(true_labels_encoded, mapped_labels, average='weighted', zero_division=0)
        }
        
        print(f"\n============K-means Clustering on {title}==========")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        # print("\nDetailed Classification Report:")
        # print(classification_report(true_labels_encoded, mapped_labels, target_names=label_encoder.classes_, zero_division=0))
        
        # Visualize clusters using t-SNE on PCA-reduced embeddings
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings_reduced)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=cluster_labels, cmap='tab20', s=20, alpha=0.8)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f"t-SNE Visualization of {title} with K-means Clusters")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.show()
        
        return metrics
    except Exception as e:
        print(f"Error in evaluate_kmeans ({title}): {e}")
        return None
    
# Evaluate K-means on raw embeddings
kmeans_raw_metrics = evaluate_kmeans(embeddings_raw, true_labels_encoded, label_encoder, 
                                    f"Raw {embed_type} Embeddings", 
                                    f"/home/saiful/ePPI_dgl/clustering/figures_paper/raw_embeddings_kmeans_tsne_{embed_type}.png")


#%% Define FCN Model
class FCNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FCNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.act4 = nn.GELU()
        self.fc5 = nn.Linear(64, num_classes)
        
        # Xavier Initialization
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, return_embeddings=False):
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.act3(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.act4(self.bn4(self.fc4(x)))
        embeddings = x
        x = self.fc5(x)
        if return_embeddings:
            return x, embeddings
        return x
# TransformerClassifier
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, nhead=4, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=nhead, dim_feedforward=512, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(128, num_classes)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, return_embeddings=False):
        x = self.input_norm(x)
        x = self.embedding(x.unsqueeze(1))
        x = self.transformer(x).squeeze(1)
        embeddings = x  # 128-dimensional embeddings
        x = self.fc(x)
        if return_embeddings:
            return x, embeddings
        return x

#%% Extract Refined Embeddings
def get_refined_embeddings(embeddings_raw, true_labels_encoded):
    try:
        X = torch.tensor(embeddings_raw, dtype=torch.float32)
        y = torch.tensor(true_labels_encoded, dtype=torch.long)
        
        print(f"Input tensor shape: {X.shape}")
        print(f"Label tensor shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Instantiate model
        input_dim = X.shape[1]
        num_classes = len(np.unique(true_labels_encoded))
        print(f"TransformerClassifier input dimension: {input_dim}, Number of classes: {num_classes}")
        model = TransformerClassifier(input_dim, num_classes)
        print("\TransformerClassifier:", model)

        
        # Train model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model.train()
        for epoch in range(600):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch % 50) == 0:
                print(f"TransformerClassifier Epoch {epoch+1}/600, Loss: {total_loss:.4f}")
        
        # Extract refined embeddings
        model.eval()
        with torch.no_grad():
            _, refined_embeddings = model(X, return_embeddings=True)
        refined_embeddings = refined_embeddings.numpy()
        print(f"Refined {embed_type} Embeddings shape: {refined_embeddings.shape}")
        return refined_embeddings
    except Exception as e:
        print(f"Error in get_refined_embeddings: {e}")
        raise
#%%Get refined embeddings
try:
    refined_embeddings = get_refined_embeddings(embeddings_raw, true_labels_encoded)
except Exception as e:
    print(f"Failed to generate Refined {embed_type} Embeddings: {e}")
    exit()


# Plot refined embeddings with true labels
plot_embeddings(refined_embeddings, true_labels_encoded, f"t-SNE Visualization of Refined {embed_type} Embeddings", 
                f"/home/saiful/ePPI_dgl/clustering/figures_paper/refined_embeddings_tsne_{embed_type}.png")

# Evaluate K-means on refined embeddings
kmeans_refined_metrics = evaluate_kmeans(refined_embeddings, true_labels_encoded, label_encoder, 
                                        f"Refined {embed_type} Embeddings", 
                                        f"/home/saiful/ePPI_dgl/clustering/figures_paper/refined_embeddings_kmeans_tsne_{embed_type}.png")

#%% Define Transformer Model
class TransformerClassifier2(nn.Module):
    def __init__(self, input_dim, num_classes, nhead=4, num_layers=2):
        super(TransformerClassifier2, self).__init__()
        self.input_norm = nn.LayerNorm(input_dim)  # Normalize input
        self.embedding = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=nhead, dim_feedforward=512, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(128, num_classes)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.input_norm(x)  # Normalize input
        x = self.embedding(x.unsqueeze(1))
        x = self.transformer(x).squeeze(1)
        x = self.fc(x)
        return x

#%% Train and Evaluate Classifiers
def evaluate_classifier(classifier, X_train, X_test, y_train, y_test, classifier_name, embedding_type, label_encoder):
    try:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        print(f"\n============{classifier_name} on {embedding_type} Embeddings==========")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print("\nDetailed Classification Report:")
        # print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
        
        return metrics
    except Exception as e:
        print(f"Error in evaluate_classifier ({classifier_name}, {embedding_type}): {e}")
        return None

def evaluate_fcn(X_train, X_test, y_train, y_test, input_dim, num_classes, classifier_name, embedding_type, label_encoder):
    try:
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Reduced batch size
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = FCNClassifier(input_dim, num_classes)
        print("\FCNClassifier:", model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00005)  # Lower learning rate
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Learning rate decay
        
        model.train()
        for epoch in range(20):  # Increased epochs
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch % 5) == 0:
                print(f"{classifier_name} Epoch {epoch+1}/5, Loss: {total_loss:.4f}")
        
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds)
                all_targets.append(y_batch)
        
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_targets = torch.cat(all_targets).cpu().numpy()
        
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
            'f1_score': f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        }
        
        print(f"\n============{classifier_name} on {embedding_type} Embeddings==========")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(all_targets, all_preds, target_names=label_encoder.classes_, zero_division=0))
        
        return metrics
    except Exception as e:
        print(f"Error in evaluate_fcn ({classifier_name}, {embedding_type}): {e}")
        return None

#%% Prepare Data for Classification
# Standardize embeddings for all methods
scaler_raw = StandardScaler()
embeddings_raw_scaled = scaler_raw.fit_transform(embeddings_raw)
scaler_refined = StandardScaler()
embeddings_refined_scaled = scaler_refined.fit_transform(refined_embeddings)

# Split data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    embeddings_raw_scaled, true_labels_encoded, test_size=0.2, random_state=42
)
X_train_refined, X_test_refined, _, _ = train_test_split(
    embeddings_refined_scaled, true_labels_encoded, test_size=0.2, random_state=42
)

# Convert to tensors for Transformer
X_train_raw_torch = torch.tensor(X_train_raw, dtype=torch.float32)
X_test_raw_torch = torch.tensor(X_test_raw, dtype=torch.float32)
X_train_refined_torch = torch.tensor(X_train_refined, dtype=torch.float32)
X_test_refined_torch = torch.tensor(X_test_refined, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.long)
y_test_torch = torch.tensor(y_test, dtype=torch.long)

#%% Evaluate scikit-learn  Classifiers
classifiers = [
    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("SVM", SVC(kernel='rbf', random_state=42)),
    ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ("Naive Bayes", GaussianNB()),
    ("Decision Tree", DecisionTreeClassifier(random_state=42, max_depth=10)),
    ("XGBoost", XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')),
    ("Logistic Regression", LogisticRegression(multi_class='multinomial', random_state=42, max_iter=1000))
]

metrics_dict = {
    "K-means_raw": kmeans_raw_metrics,
    "K-means_refined": kmeans_refined_metrics
}

# Evaluate scikit-learn classifiers
for name, clf in classifiers:
    metrics_dict[f"{name}_raw"] = evaluate_classifier(clf, X_train_raw, X_test_raw, y_train, y_test, name, "Raw", label_encoder)
    metrics_dict[f"{name}_refined"] = evaluate_classifier(clf, X_train_refined, X_test_refined, y_train, y_test, name, "Refined", label_encoder)
#%% Evaluate FCN
num_classes = len(np.unique(true_labels_encoded))
metrics_dict["FCN_raw"] = evaluate_fcn(
    X_train_raw_torch, X_test_raw_torch, y_train_torch, y_test_torch, 
    embeddings_raw_scaled.shape[1], num_classes, "FCN", "Raw", label_encoder
)
metrics_dict["FCN_refined"] = evaluate_fcn(
    X_train_refined_torch, X_test_refined_torch, y_train_torch, y_test_torch, 
    embeddings_refined_scaled.shape[1], num_classes, "FCN", "Refined", label_encoder
)

#%% Comparison Table
def print_comparison_table(metrics_dict):
    print("\n============ Comparison Table ==========")
    for name, metrics in metrics_dict.items():
        if metrics is not None:
            print(f"{name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
        else:
            print(f"{name}: Failed to compute metrics")

# Print comparison table
print_comparison_table(metrics_dict)