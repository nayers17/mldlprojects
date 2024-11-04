import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Check the first few rows to understand the data
print(data.head())

# Select features for clustering (e.g., 'Annual Income (k$)' and 'Spending Score (1-100)')
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Create a KMeans model with 3 clusters (you can adjust this number based on your analysis)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Plotting the clusters
# Save the plot instead of showing it directly
plt.figure(figsize=(10, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.colorbar(label='Cluster')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation')

# Save the plot as an image file
plt.savefig('customer_segmentation_plot.png')
print("Plot saved as 'customer_segmentation_plot.png'")
