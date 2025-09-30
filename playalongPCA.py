import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
data = np.array([
    [2.5, 2.325,  1.8],
    [0.5, 0.325,  -0.2],
    [2.2, 2.120, -0.7],
    [1.9, 1.065,  2.1],
    [3.1, 2.735,  -0.1],
    [2.3, 2.005,  1.5]
])

# first pass to pandas and plot as-is using seaborn
df = pd.DataFrame(data, columns=['MFCC1', 'Flux', 'ZCR'])

# . Plot the raw features in 3 dimensions using seaborn’s scatterplot() function:

plt.figure()
# First arg is the data (full pd.DataFrame)
# Pass one feature (column) as ‘x’, one as ‘y’, and one to the ‘size’ or ‘hue’ variable
sns.scatterplot(data=df, x='MFCC1', y='Flux', size='ZCR', sizes=(20, 200), alpha=0.5)
#show plot with title and labels
plt.title('Raw Audio Features')
# plt.show


# Normalize the data using z-score 
from scipy.stats import zscore
data_z = zscore(data)

# Create a new DataFrame with the normalized data
df_z = pd.DataFrame(data_z, columns=['MFCC1', 'Flux', 'ZCR'])

# Show the normalized DataFrame
print(df_z)
plt.figure()
# Plot the normalized features
sns.scatterplot(data=df_z, x='MFCC1', y='Flux', size='ZCR', sizes=(20, 200), alpha=0.5)
#show plot
plt.title('Normalized Audio Features (Z-score)')
# plt.show()  

#compute covariance matrix using np.cov()
cov_matrix = np.cov(data_z, rowvar=False)
#pass covariance matric to np,lingalg.eig and sort
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
#sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
#compute explained variance ratio by dividing each eigenvalue by total sum
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
#project data to PC space by computing dot product of the matric of z score data with the eighen vectors
data_pca = np.dot(data_z, eigenvectors)
# Create a new DataFrame with the PCA-transformed data
df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2', 'PC3'])
# Show the PCA-transformed DataFrame
print(df_pca)

# Combine PC scores with z-scored features for mixed plotting needs
combined_df = pd.concat([
    df_z.add_prefix('zscore_'),
    df_pca
], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot PC1 against z-scored versions of the original features
sns.scatterplot(
    data=combined_df,
    x='PC1',
    y='zscore_Flux',
    size='zscore_ZCR',
    hue='zscore_MFCC1',
    sizes=(20, 200),
    alpha=0.5,
    ax=axes[0]
)
axes[0].set_title('PC1 with Z-scored Features')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('Z-scored Flux')

# Plot PC2 to inspect the PCA-transformed feature space directly
sns.scatterplot(
    data=df_pca,
    x='PC2',
    y='PC1',
    size='PC3',
    sizes=(20, 200),
    alpha=0.5,
    ax=axes[1]
)
axes[1].set_title('PC2 with PCA-Transformed Features')
axes[1].set_xlabel('PC2')
axes[1].set_ylabel('PC1')

plt.tight_layout()
plt.show()

print("yay")