#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 02:45:20 2025

@author: saiful
"""

import pandas as pd

# Load the CSV file
file_path = "/home/saiful/ePPI_dgl/clustering/protein_families23k.csv"
df_protein_families = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df_protein_families.head())

# Select only the desired columns
df_selected = df_protein_families[['Protein_ID', 'Family']]
print(df_selected.head())

# Filter out rows where 'Family' contains "No SIMILARITY comment found"
df_filtered = df_selected[df_selected['Family'] != "No SIMILARITY comment found"]
df_filtered = df_filtered[df_filtered['Family'] != "No family label found"]

# Count the number of unique string values in the 'Family' column
unique_family_count = df_filtered['Family'].nunique()

# Display the count
print(f"Number of unique families: {unique_family_count}")

# filter the DataFrame for families that occur more than 60 times#%%

# Count the frequency of each family
family_frequencies = df_filtered['Family'].value_counts()

# Display the frequencies
print(family_frequencies)

# Count the frequency of each family
family_frequencies = df_filtered['Family'].value_counts()

# Filter families with frequency greater than 50
frequent_families = family_frequencies[family_frequencies > 60].index

# Filter the DataFrame to include only these families
df_filtered_more_than_50 = df_filtered[df_filtered['Family'].isin(frequent_families)]

# Display the resulting DataFrame
print(df_filtered_more_than_50.head())

# Count the frequency of each family
family_frequencies_50 = df_filtered_more_than_50['Family'].value_counts()

# Display the frequencies
print(family_frequencies_50)

#%% get corresponding alphafold
import pandas as pd
import ast
import numpy as np
# Load AlphaFold embeddings
alphafold_embeddings_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/eppi_alphafold_aggregated_embeddings.csv"
df_alphafold = pd.read_csv(alphafold_embeddings_path)


# Convert string representations of lists to actual lists
df_alphafold['aggregated_features'] = df_alphafold['aggregated_features'].apply(ast.literal_eval)
# Randomly select 2000 rows
# df_alphafold = df_alphafold.sample(n=200, random_state=42)
# Convert aggregated_features to a numpy array
embeddings = np.array(df_alphafold['aggregated_features'].tolist())

# Merge the two DataFrames on protein IDs (inner join)
merged_df = pd.merge(df_alphafold, 
                     df_filtered_more_than_50, 
                     left_on='protein_id', 
                     right_on='Protein_ID', 
                     how='inner')

# Display the merged DataFrame
print(merged_df.head())
#%% knn 1
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#  Preprocess the data
# Convert aggregated_features from string to list of floats
# merged_df['aggregated_features'] = merged_df['aggregated_features'].apply(eval)

embeddings = np.array(merged_df['aggregated_features'].tolist())
true_labels = merged_df['Family']

# # Use PCA to reduce embeddings to 2D for visualization
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

#  Clustering
# Use k-means clustering with k = number of unique families
k = len(true_labels.unique())  # Number of clusters = number of unique families
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

#  Evaluate Clustering Quality
ari = adjusted_rand_score(true_labels, cluster_labels)
nmi = normalized_mutual_info_score(true_labels, cluster_labels)
# f1 = f1_score(true_labels, cluster_labels, average='weighted')

print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
# print(f"Weighted F1-score: {f1:.4f}")


#  Visualize Clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=cluster_labels, cmap='tab20', s=20, alpha=0.8)
plt.colorbar(scatter, label='Cluster')
plt.title("flag 1.1 Protein Clusters in PCA Space")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
#%% knn2 ==> working well
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt

# Preprocess the data
# Convert 'aggregated_features' from string to list of floats (if necessary)
# merged_df['aggregated_features'] = merged_df['aggregated_features'].apply(eval)

# Extract embeddings and true labels
embeddings = np.array(merged_df['aggregated_features'].tolist())
true_labels = merged_df['Family']

# #  Normalize/Scale the Data
# scaler = StandardScaler()
# embeddings_scaled = scaler.fit_transform(embeddings)


embeddings_scaled = embeddings
#  Dimensionality Reduction
# Use PCA to retain 95% of the variance
pca = PCA(n_components=0.95)
embeddings_reduced = pca.fit_transform(embeddings_scaled)

# Use t-SNE for 2D visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings_reduced)

#  Clustering
# Determine the number of clusters (k) based on the number of unique families
k = len(true_labels.unique())
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings_reduced)

#  Evaluate Clustering Quality
ari = adjusted_rand_score(true_labels, cluster_labels)
nmi = normalized_mutual_info_score(true_labels, cluster_labels)
silhouette = silhouette_score(embeddings_reduced, cluster_labels)

print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

# Step 6: Visualize Clusters
# Plot the clusters with t-SNE
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=cluster_labels, cmap='tab20', s=20, alpha=0.8
)
plt.colorbar(scatter, label='Cluster')
plt.title("Protein Clusters in t-SNE Space")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.savefig("/home/saiful/ePPI_dgl/clustering/figures/flag_1.2_knn_protein_cluster.png", dpi=400, bbox_inches='tight')

plt.show()



#%% supervised classification  with Random Forest  ==> working well 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


# Step 1: Prepare Data
# Convert 'aggregated_features' into a numpy array
embeddings = np.array(merged_df['aggregated_features'].tolist())
true_labels = merged_df['Family']

# Encode family labels as integers
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)

# # Standardize the features using StandardScaler
# scaler = StandardScaler()
# embeddings = scaler.fit_transform(embeddings) 

# Step 2: Split Data
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, true_labels_encoded, test_size=0.2, random_state=42
)

# Step 3: Train a Classifier
# Use Random Forest Classifier for supervised learning
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 4: Evaluate Classifier
# Predict labels for the test set
y_pred = rf_classifier.predict(X_test)

# Compute overall precision, recall, and F1-score (weighted average)
overall_precision = precision_score(y_test, y_pred, average='weighted')
overall_recall = recall_score(y_test, y_pred, average='weighted')
overall_f1 = f1_score(y_test, y_pred, average='weighted')

# Print the metrics
# Compute accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print(f"Accuracy: {accuracy:.4f}")
print(f"Overall Precision: {overall_precision:.4f}")
print(f"Overall Recall: {overall_recall:.4f}")
print(f"Overall F1-Score: {overall_f1:.4f}")


#%% using fcn  flag 1.7
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Prepare Data
# Convert 'aggregated_features' into a numpy array
embeddings = np.array(merged_df['aggregated_features'].tolist())
true_labels = merged_df['Family']

# Encode family labels as integers
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)

# Convert data to PyTorch tensors
X = torch.tensor(embeddings, dtype=torch.float32)
y = torch.tensor(true_labels_encoded, dtype=torch.long)

# Step 2: Split Data
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataLoaders
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 3: Define the Fully Connected Neural Network
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Instantiate the model
input_dim = X.shape[1]
num_classes = len(np.unique(true_labels_encoded))
model = FullyConnectedNN(input_dim, num_classes)

# Step 4: Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Step 5: Train the Model
num_epochs = 300
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if(epoch% 50==0):
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Step 6: Evaluate the Model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y_batch.numpy())

# Compute metrics
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# Step 7: Use the Model for Refinement
# Predict labels for the entire dataset
with torch.no_grad():
    outputs = model(X)
    _, refined_labels = torch.max(outputs, 1)

# Step 8: Visualize Results
# Use t-SNE for visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

# Plot the t-SNE visualization with refined labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=refined_labels.numpy(), cmap='tab10', s=10, alpha=0.8
)
plt.colorbar(scatter, label="Refined Labels (Neural Network)")
plt.title("flag 1.7 t-SNE Visualization with Refined Family Labels")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

# Step 9: Compute Overall Precision, Recall, and F1-Score
overall_precision = precision_score(all_labels, all_preds, average='weighted')
overall_recall = recall_score(all_labels, all_preds, average='weighted')
overall_f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Overall Precision: {overall_precision:.4f}")
print(f"Overall Recall: {overall_recall:.4f}")
print(f"Overall F1-Score: {overall_f1:.4f}")


#%% flag 1.8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Prepare Data
# Convert 'aggregated_features' into a numpy array
embeddings = np.array(merged_df['aggregated_features'].tolist())
true_labels = merged_df['Family']
unique_labels = true_labels.unique()
print(unique_labels)

# Encode family labels as integers
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)

# Convert data to PyTorch tensors
X = torch.tensor(embeddings, dtype=torch.float32)
y = torch.tensor(true_labels_encoded, dtype=torch.long)

# Step 2: Split Data
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataLoaders
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 3: Define the Fully Connected Neural Network
class FullyConnectedNN2(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)  # Output of this layer will be used as refined embeddings
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x, return_embeddings=False):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # Embeddings are extracted here
        embeddings = x
        x = self.relu(x)
        x = self.fc3(x)
        if return_embeddings:
            return x, embeddings  # Return both logits and embeddings
        return x
    
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FullyConnectedNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)  # Increased neurons
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization
        self.act1 = nn.GELU()  # GELU activation for smoother gradients
        self.dropout1 = nn.Dropout(0.3)  # Dropout for regularization
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(128, 64)  # Extract refined embeddings
        self.bn4 = nn.BatchNorm1d(64)
        self.act4 = nn.GELU()
        
        self.fc5 = nn.Linear(64, num_classes)  # Output layer

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
        
        x = self.act4(self.bn4(self.fc4(x)))  # Extract refined embeddings
        embeddings = x  # Save refined embeddings
        
        x = self.fc5(x)  # Output layer
        
        if return_embeddings:
            return x, embeddings  # Return both logits and refined embeddings
        return x

# Instantiate the model
input_dim = X.shape[1]
num_classes = len(np.unique(true_labels_encoded))
model = FullyConnectedNN(input_dim, num_classes)

# Step 4: Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Step 5: Train the Model
num_epochs = 300
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)  # Only classification logits needed during training
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if(epoch % 50)==0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Step 6: Extract Refined Embeddings
model.eval()
with torch.no_grad():
    _, refined_embeddings = model(X, return_embeddings=True)  # Get refined embeddings

refined_embeddings = refined_embeddings.numpy()  # Convert to numpy array

# Step 7: Cluster Refined Embeddings
kmeans = KMeans(n_clusters=num_classes, random_state=42)
cluster_labels = kmeans.fit_predict(refined_embeddings)

# # Step 8: Evaluate Clustering
# # Map cluster labels to true labels based on majority voting
cluster_to_label = {}
for cluster in np.unique(cluster_labels):
    cluster_indices = np.where(cluster_labels == cluster)[0]
    true_labels_cluster = true_labels_encoded[cluster_indices]
    majority_label = np.bincount(true_labels_cluster).argmax()
    cluster_to_label[cluster] = majority_label

mapped_labels =cluster_labels
# Map predicted cluster labels to original family labels
mapped_labels = np.array([cluster_to_label[cluster] for cluster in cluster_labels])

# Evaluate clustering quality
print("\nClustering Classification Report:")
print(classification_report(true_labels_encoded, mapped_labels, target_names=label_encoder.classes_))

# Step 9: Visualize Clusters
# Use t-SNE for visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_tsne = tsne.fit_transform(refined_embeddings)

# Plot the t-SNE visualization with cluster labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=cluster_labels, cmap='tab20', s=20, alpha=0.8
)
plt.colorbar(scatter, label="Cluster Labels")
plt.title(" t-SNE Visualization of Refined Embeddings") #flag 1.8
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
# plt.savefig("/home/saiful/ePPI_dgl/clustering/figures/flag_1.8_Refined_Embeddings.png", dpi=400, bbox_inches='tight')
plt.show()



# Step 10: Compute Overall Precision, Recall, and F1-Score
overall_precision = precision_score(true_labels_encoded, mapped_labels, average='weighted')
overall_recall = recall_score(true_labels_encoded, mapped_labels, average='weighted')
overall_f1 = f1_score(true_labels_encoded, mapped_labels, average='weighted')


# Evaluate clustering quality
print("\nClustering Classification Report:")
from sklearn.metrics import classification_report

print(classification_report(true_labels_encoded, mapped_labels, target_names=label_encoder.classes_))

# Compute Classification Accuracy
classification_accuracy = np.mean(mapped_labels == true_labels_encoded)
print(f"Classification Accuracy: {classification_accuracy:.4f}")
print(f"Overall Precision: {overall_precision:.4f}")
print(f"Overall Recall: {overall_recall:.4f}")
print(f"Overall F1-Score: {overall_f1:.4f}")