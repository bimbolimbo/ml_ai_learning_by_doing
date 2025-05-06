import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd # Added for DataFrame handling
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. Load Iris Data ---
iris = px.data.iris()
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_labels = iris['species'] # For coloring plots later
feature_names = X.columns.tolist() # Store original feature names if needed

# --- 2. Preprocess Data ---
# PCA is sensitive to scale, so standardize the data (mean=0, variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# data_mean = scaler.mean_ # Mean of original features (less relevant now)


# --- 3. Perform PCA ---
n_components = 3 # Reduce dimensionality to 3 for visualization
pca = PCA(n_components=n_components)
# X_pca contains the coordinates of the data projected onto the principal components
X_pca = pca.fit_transform(X_scaled)

# --- 4. Extract PCA Results ---
explained_variance_ratio = pca.explained_variance_ratio_
components = pca.components_ # Eigenvectors in terms of original scaled features
# The origin for the PCA projection space is [0,0,0]
pca_origin = [0, 0, 0]

# --- 5. Create Interactive Plots ---

# Create subplots: 1 row, 2 cols. First for 3D view, second for Scree + 2D Projection
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'scatter3d', 'rowspan': 2}, {'type': 'xy'}],
           [None, {'type': 'xy'}]],
    subplot_titles=("Data Projected onto First 3 Principal Components",
                    "Scree Plot (Variance Explained by PCs)",
                    "Data Projected onto First 2 Principal Components")
)

# --- Plot 1: 3D Scatter Plot of PCA Projection ---

# Map species names to integer IDs for coloring
species_map = {name: i for i, name in enumerate(y_labels.unique())}
species_ids = y_labels.map(species_map)

# Add data points projected onto the first 3 PCs
fig.add_trace(
    go.Scatter3d(
        x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], # Use PC coordinates
        mode='markers',
        marker=dict(
            size=5,
            color=species_ids, # Color by species ID
            colorscale='Viridis', # Choose a colorscale
            opacity=0.7,
            colorbar=dict(title='Species', tickvals=list(species_map.values()), ticktext=list(species_map.keys())) # Add colorbar legend
        ),
        customdata=y_labels, # Store species names for hover
        hoverinfo='text', # Use custom hover text
        hovertext=[f'Species: {y_labels.iloc[i]}<br>PC1: {X_pca[i,0]:.2f}<br>PC2: {X_pca[i,1]:.2f}<br>PC3: {X_pca[i,2]:.2f}' for i in range(len(X_pca))],
        name='Projected Data (PCs 1-3)'
    ),
    row=1, col=1
)

# Note: PC Axes lines are removed as the axes of this plot ARE PC1, PC2, PC3.

# --- Plot 2: Scree Plot ---
pc_indices = [f'PC{i+1}' for i in range(n_components)]
fig.add_trace(
    go.Bar(
        x=pc_indices,
        y=explained_variance_ratio,
        name='Explained Variance Ratio',
        hovertext=[f'{ratio:.3f}' for ratio in explained_variance_ratio]
    ),
    row=1, col=2
)

# --- Plot 3: 2D Projection ---
# Plot the data transformed (projected) onto the first two PCs
fig.add_trace(
    go.Scatter(
        x=X_pca[:, 0], # Coordinates on PC1
        y=X_pca[:, 1], # Coordinates on PC2
        mode='markers',
        marker=dict(
           color=species_ids, # Color by species ID
           colorscale='Viridis',
           opacity=0.8
        ),
        customdata=y_labels, # Store species names for hover
        hoverinfo='text', # Use custom hover text
        hovertext=[f'Species: {y_labels.iloc[i]}<br>PC1: {X_pca[i,0]:.2f}<br>PC2: {X_pca[i,1]:.2f}' for i in range(len(X_pca))],
        name='Projected Data (PC1 vs PC2)'
    ),
    row=2, col=2
)

# --- Update Layout ---
fig.update_layout(
    title_text='Interactive Principal Component Analysis (PCA) on Iris Dataset',
    height=800, # Adjust height as needed
    # Settings for the 3D plot (now showing PCs)
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3',
        aspectratio=dict(x=1, y=1, z=1), # Ensure axes are equally scaled
        camera_eye=dict(x=1.2, y=1.2, z=1.2) # Initial camera view
    ),
    # Settings for the Scree Plot (subplot at row=1, col=2 -> axes x2, y2)
    xaxis2=dict(title='Principal Component'),
    yaxis2=dict(title='Explained Variance Ratio'),
    # Settings for the 2D Projection Plot (subplot at row=2, col=2 -> axes x3, y3)
    xaxis3=dict(title='Principal Component 1'),
    yaxis3=dict(title='Principal Component 2'),
    margin=dict(l=0, r=0, b=0, t=50),
    hovermode='closest', # Show hover for nearest point
    showlegend=False # Hide default legend as colorbar is used
)

# --- Show Plot ---
fig.show()

# To save as an HTML file:
# fig.write_html("interactive_pca_visualization_iris.html")