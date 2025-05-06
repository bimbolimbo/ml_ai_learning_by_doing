import dash
from dash import dcc, html, Input, Output, State, no_update, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json # For storing data in dcc.Store

# --- 1. Load Initial Data ---
initial_iris = px.data.iris()
initial_features = initial_iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
initial_labels = initial_iris['species']
feature_names = initial_features.columns.tolist()
original_species = initial_labels.unique().tolist() # Get list of original species
# Define the labels available for adding/relabeling
add_point_label_options = original_species + ['Added Point']

# Store initial data in the required JSON format for the store
initial_store_data = {
    'features': initial_features.to_json(orient='split'),
    'labels': initial_labels.to_json(orient='split')
}

# --- Helper Function to Create SPLOM Figure --- NEW
def create_splom_figure(df_features, df_labels):
    if df_features.empty:
        return go.Figure().update_layout(title_text="No data for SPLOM")

    # --- Map Labels to Colors ---
    unique_labels = df_labels.unique()
    color_map = {label: i for i, label in enumerate(unique_labels)}
    label_ids = df_labels.map(color_map)
    colorscale = px.colors.qualitative.Plotly

    # --- Create SPLOM Trace ---
    splom_dimensions = [dict(label=col, values=df_features[col]) for col in df_features.columns]
    splom_trace = go.Splom(
        dimensions=splom_dimensions,
        showupperhalf=False,
        diagonal_visible=False,
        marker=dict(
            color=label_ids,
            size=5,
            coloraxis="coloraxis", # Use a coloraxis name
            line_color='white',
            line_width=0.5,
        ),
        customdata=np.arange(len(df_labels)), # Pass index for potential future use
        hoverinfo='text',
        hovertext=[f'Index: {i}<br>Label: {df_labels.iloc[i]}' for i in range(len(df_labels))]
    )

    # --- Create Figure with SPLOM ---
    fig = go.Figure(data=[splom_trace])

    # --- Layout Updates for SPLOM Figure ---
    fig.update_layout(
        title_text='Original Feature Pairwise Scatter Plots (SPLOM)',
        height=600,
        width=600,
        hovermode='closest',
        coloraxis=dict( # Define the color axis
            colorscale=colorscale,
            cmin=0,
            cmax=max(0, len(unique_labels) - 1),
            colorbar=dict(
                title="Label",
                tickvals=list(color_map.values()),
                ticktext=list(color_map.keys())
            )
        ),
        showlegend=False,
        dragmode='select' # Optional: change drag mode
    )
    # Suppress axis titles within SPLOM for cleaner look
    fig.update_xaxes(showticklabels=False, title_text="")
    fig.update_yaxes(showticklabels=False, title_text="")

    return fig

# --- Helper Function to Perform PCA and Create Figure (PCA plots only) --- MODIFIED
def create_pca_figure(df_features, df_labels):
    if df_features.empty or len(df_features) < 2:
        fig = go.Figure()
        fig.update_layout(title_text="Not enough data points for PCA (minimum 2 required)")
        return fig

    # --- Preprocess for PCA ---
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(df_features)
    except ValueError:
        X_scaled = df_features.to_numpy()

    # --- PCA ---
    n_components = min(3, X_scaled.shape[0], X_scaled.shape[1])
    if n_components < 1:
        fig = go.Figure()
        fig.update_layout(title_text="Not enough features/samples for PCA")
        return fig

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_

    # --- Padding for PCA components (if needed) ---
    X_pca_vis = X_pca.copy()
    if X_pca_vis.shape[1] < 3:
        X_pca_padded = np.zeros((X_pca_vis.shape[0], 3))
        X_pca_padded[:, :X_pca_vis.shape[1]] = X_pca_vis
        X_pca_vis = X_pca_padded
    if X_pca_vis.shape[1] < 2:
        X_pca_padded = np.zeros((X_pca_vis.shape[0], max(2, X_pca_vis.shape[1])))
        X_pca_padded[:, :X_pca_vis.shape[1]] = X_pca_vis
        X_pca_vis = X_pca_padded

    # --- Create Figure --- 1x3 layout for PCA plots
    fig = make_subplots(
        rows=1, cols=3, # Adjusted layout
        specs=[[{'type': 'xy'}, {'type': 'scatter3d'}, {'type': 'xy'}]], # Scree, 3D, 2D
        subplot_titles=("PCA Scree Plot",
                        "Data Projected onto First 3 PCs (Click to Remove)",
                        "Data Projected onto First 2 PCs (Click to Remove)"),
        column_widths=[0.2, 0.5, 0.3] # Adjust relative widths
    )

    # --- Map Labels to Colors ---
    unique_labels = df_labels.unique()
    color_map = {label: i for i, label in enumerate(unique_labels)}
    label_ids = df_labels.map(color_map)
    colorscale = px.colors.qualitative.Plotly

    # --- Plot 1: Scree Plot --- Row 1, Col 1
    pc_indices = [f'PC{i+1}' for i in range(len(explained_variance_ratio))]
    fig.add_trace(
        go.Bar(
            x=pc_indices,
            y=explained_variance_ratio,
            name='Explained Variance'
        ),
        row=1, col=1 # New Position
    )

    # --- Plot 2: 3D PCA Projection --- Row 1, Col 2
    fig.add_trace(
        go.Scatter3d(
            x=X_pca_vis[:, 0], y=X_pca_vis[:, 1], z=X_pca_vis[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=label_ids,
                coloraxis="coloraxis", # Use a coloraxis name
                opacity=0.8,
                line=dict(width=1, color=['black' if l == 'Added Point' else 'rgba(0,0,0,0)' for l in df_labels])
            ),
            customdata=np.arange(len(df_labels)), # Pass index
            hoverinfo='text',
            hovertext=[f'Index: {i}<br>Label: {df_labels.iloc[i]}<br>PC1: {X_pca_vis[i,0]:.2f}<br>PC2: {X_pca_vis[i,1]:.2f}<br>PC3: {X_pca_vis[i,2]:.2f}' for i in range(len(X_pca_vis))],
            name='PCA 3D Projection'
        ),
        row=1, col=2 # New Position
    )

    # --- Plot 3: 2D PCA Projection --- Row 1, Col 3
    fig.add_trace(
        go.Scatter(
            x=X_pca_vis[:, 0], y=X_pca_vis[:, 1],
            mode='markers',
            marker=dict(
               color=label_ids,
               coloraxis="coloraxis", # Use a coloraxis name
               opacity=0.8,
               size=6,
               line=dict(width=1, color=['black' if l == 'Added Point' else 'rgba(0,0,0,0)' for l in df_labels])
            ),
            customdata=np.arange(len(df_labels)), # Pass index
            hoverinfo='text',
            hovertext=[f'Index: {i}<br>Label: {df_labels.iloc[i]}<br>PC1: {X_pca_vis[i,0]:.2f}<br>PC2: {X_pca_vis[i,1]:.2f}' for i in range(len(X_pca_vis))],
            name='PCA 2D Projection'
        ),
        row=1, col=3 # New Position
    )

    # --- Layout Updates for PCA Figure ---
    fig.update_layout(
        title_text='Principal Component Analysis Results', # Simplified title
        height=500, # Adjusted height
        # Settings for 3D PCA Plot (scene1)
        scene1=dict(
            xaxis_title='PC 1', yaxis_title='PC 2', zaxis_title='PC 3',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        # Settings for Scree Plot (axes 1)
        xaxis1=dict(title='Principal Component'),
        yaxis1=dict(title='Explained Variance Ratio'),
        # Settings for 2D PCA Plot (axes 3)
        xaxis3=dict(title='PC 1'),
        yaxis3=dict(title='PC 2'),
        margin=dict(l=40, r=40, b=40, t=50),
        hovermode='closest',
        coloraxis=dict( # Define the color axis
            colorscale=colorscale,
            cmin=0,
            cmax=max(0, len(unique_labels) - 1),
            colorbar=dict(
                title="Label",
                tickvals=list(color_map.values()),
                ticktext=list(color_map.keys())
            )
        ),
        showlegend=False
    )

    return fig

# --- Dash App Initialization ---
app = dash.Dash(__name__)
server = app.server # Expose server for deployment

# --- App Layout --- MODIFIED
app.layout = html.Div([
    html.H1("Interactive PCA Visualization"),
    html.P("Explore original features (left) and PCA results (right). Add/remove/relabel points using controls below."),

    # Data Store
    dcc.Store(id='pca-data-store', storage_type='session', data=initial_store_data),

    # Arrange Graphs Side-by-Side
    html.Div([
        dcc.Graph(id='splom-graph', style={'display': 'inline-block', 'width': '49%'}),
        dcc.Graph(id='pca-graph', style={'display': 'inline-block', 'width': '49%'})
    ]),

    # --- Input Controls --- #
    html.Div([
        # --- Add Point Section --- #
        html.Div([
            html.H4("Add a New Data Point:"),
            html.Div([
                html.Label("Sepal L:"),
                dcc.Input(id='input-sepal-length', type='number', value=5.0, step=0.1, style={'width': '80px'}),
                html.Label("Sepal W:"),
                dcc.Input(id='input-sepal-width', type='number', value=3.5, step=0.1, style={'width': '80px'}),
                html.Label("Petal L:"),
                dcc.Input(id='input-petal-length', type='number', value=1.4, step=0.1, style={'width': '80px'}),
                html.Label("Petal W:"),
                dcc.Input(id='input-petal-width', type='number', value=0.2, step=0.1, style={'width': '80px'}),
                html.Label("Label:"),
                dcc.Dropdown(
                    id='add-point-label-dropdown',
                    options=[{'label': s, 'value': s} for s in add_point_label_options],
                    value='Added Point',
                    clearable=False,
                    style={'width': '150px'}
                ),
            ], style={'display': 'flex', 'gap': '10px', 'alignItems': 'center', 'flexWrap': 'wrap'}),
            html.Button('Add Point', id='add-point-button', n_clicks=0, style={'marginTop': '10px'})
        ], style={'marginBottom': '20px', 'padding': '15px', 'border': '1px solid #ccc', 'borderRadius': '5px'}),

        # --- Relabel Point Section --- #
        html.Div([
            html.H4("Relabel a Point:"),
            html.Div([
                html.Label("Point Index:"),
                dcc.Input(id='relabel-index-input', type='number', placeholder='Enter index...', min=0, step=1, style={'width': '100px'}),
                html.Label("New Label:"),
                dcc.Dropdown(
                    id='relabel-label-dropdown',
                    options=[{'label': s, 'value': s} for s in add_point_label_options],
                    placeholder="Select label...",
                    style={'width': '150px'}
                ),
            ], style={'display': 'flex', 'gap': '10px', 'alignItems': 'center', 'flexWrap': 'wrap'}),
            html.Button('Relabel Point', id='relabel-point-button', n_clicks=0, style={'marginTop': '10px'})
        ], style={'marginBottom': '20px', 'padding': '15px', 'border': '1px solid #ccc', 'borderRadius': '5px'}),
    ]), # End of Input Controls Div
])

# --- Callback to Add Point to Store ---
@app.callback(
    Output('pca-data-store', 'data', allow_duplicate=True),
    Input('add-point-button', 'n_clicks'),
    State('input-sepal-length', 'value'),
    State('input-sepal-width', 'value'),
    State('input-petal-length', 'value'),
    State('input-petal-width', 'value'),
    State('add-point-label-dropdown', 'value'),
    State('pca-data-store', 'data'),
    prevent_initial_call=True
)
def add_point(n_clicks, sl, sw, pl, pw, selected_label, stored_data):
    if n_clicks > 0 and all(v is not None for v in [sl, sw, pl, pw, selected_label]):
        features_df = pd.read_json(stored_data['features'], orient='split')
        labels_series = pd.read_json(stored_data['labels'], orient='split', typ='series')
        labels_series.name = 'species'
        new_point_features = pd.DataFrame([[sl, sw, pl, pw]], columns=feature_names)
        new_point_label = pd.Series([selected_label], name='species')
        features_df = pd.concat([features_df, new_point_features], ignore_index=True)
        labels_series = pd.concat([labels_series, new_point_label], ignore_index=True)
        updated_data = {
            'features': features_df.to_json(orient='split'),
            'labels': labels_series.to_json(orient='split')
        }
        return updated_data
    elif n_clicks > 0:
        print("Add Point Error: Ensure all feature values and a label are provided.")
    return no_update

# --- Callback to Remove Point from Store --- # (Still listens only to pca-graph)
@app.callback(
    Output('pca-data-store', 'data', allow_duplicate=True),
    Input('pca-graph', 'clickData'), # Only listening to clicks on PCA graph for now
    State('pca-data-store', 'data'),
    prevent_initial_call=True
)
def remove_point(clickData, stored_data):
    if clickData is None or not clickData.get('points'):
        return no_update
    curve_number = clickData['points'][0].get('curveNumber', 0)
    # Determine which plot within the PCA figure was clicked if needed (e.g., curveNumber mapping)
    # For now, assume click on 3D or 2D PCA plot is sufficient
    print(f"Clicked PCA graph curve number: {curve_number}")
    try:
        point_index = clickData['points'][0]['customdata']
        point_index = int(point_index)
    except (KeyError, IndexError, TypeError):
        print("Error extracting point index from PCA clickData:", clickData)
        return no_update
    features_df = pd.read_json(stored_data['features'], orient='split')
    labels_series = pd.read_json(stored_data['labels'], orient='split', typ='series')
    labels_series.name = 'species'
    if 0 <= point_index < len(features_df):
        print(f"Removing point at index: {point_index}")
        features_df = features_df.drop(index=point_index).reset_index(drop=True)
        labels_series = labels_series.drop(index=point_index).reset_index(drop=True)
        updated_data = {
            'features': features_df.to_json(orient='split'),
            'labels': labels_series.to_json(orient='split')
        }
        return updated_data
    else:
        print(f"Index {point_index} out of bounds for DataFrame length {len(features_df)}")
        return no_update

# --- Callback to Relabel Point in Store ---
@app.callback(
    Output('pca-data-store', 'data', allow_duplicate=True),
    Input('relabel-point-button', 'n_clicks'),
    State('relabel-index-input', 'value'),
    State('relabel-label-dropdown', 'value'),
    State('pca-data-store', 'data'),
    prevent_initial_call=True
)
def relabel_point(n_clicks, point_index, new_label, stored_data):
    if n_clicks is None or n_clicks == 0:
        return no_update
    if point_index is None or new_label is None:
        print("Relabel Error: Index or new label not provided.")
        return no_update
    try:
        point_index = int(point_index)
    except (ValueError, TypeError):
        print(f"Relabel Error: Invalid index '{point_index}'. Must be an integer.")
        return no_update
    features_df = pd.read_json(stored_data['features'], orient='split')
    labels_series = pd.read_json(stored_data['labels'], orient='split', typ='series')
    labels_series.name = 'species'
    if 0 <= point_index < len(features_df):
        print(f"Relabeling point at index: {point_index} to '{new_label}'")
        labels_series.iloc[point_index] = new_label
        updated_data = {
            'features': features_df.to_json(orient='split'),
            'labels': labels_series.to_json(orient='split')
        }
        return updated_data
    else:
        print(f"Relabel Error: Index {point_index} out of bounds for DataFrame length {len(features_df)}")
        return no_update

# --- Callback to Update BOTH Graphs from Store --- MODIFIED
@app.callback(
    Output('splom-graph', 'figure'),
    Output('pca-graph', 'figure'),
    Input('pca-data-store', 'data') # Triggered when store changes
)
def update_graphs(stored_data):
    if stored_data is None:
        empty_fig = go.Figure().update_layout(title_text="No data available")
        return empty_fig, empty_fig

    # Load data from store
    features_df = pd.read_json(stored_data['features'], orient='split')
    labels_series = pd.read_json(stored_data['labels'], orient='split', typ='series')
    labels_series.name = 'species'

    # Create the figures using the helper functions
    # Pass original features to SPLOM
    splom_fig = create_splom_figure(features_df, labels_series)
    # Pass potentially scaled features (if PCA needed scaling) to PCA plots
    pca_fig = create_pca_figure(features_df, labels_series)

    return splom_fig, pca_fig

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True) 