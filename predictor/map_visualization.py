import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
import os
import numpy as np

def create_rwanda_district_map(df):
    """
    Create a Rwanda map showing vehicle client distribution by district with proper boundaries using GeoJSON.
    Each district has a unique color, district name label, and client count displayed.
    """
    # Count clients per district
    district_counts = df['district'].value_counts().reset_index()
    district_counts.columns = ['district', 'client_count']
    
    # Load GeoJSON file
    geojson_path = os.path.join('dummy-data', 'rwanda_districts.geojson')
    with open(geojson_path, 'r', encoding='utf-8') as f:
        rwanda_geojson = json.load(f)
    
    # Generate unique colors for each district
    num_districts = len(district_counts)
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    district_colors = {district: colors[i % len(colors)] for i, district in enumerate(district_counts['district'])}
    district_counts['color'] = district_counts['district'].map(district_colors)
    
    # Calculate centroids for district labels
    centroids = {}
    for feature in rwanda_geojson['features']:
        district_name = feature['properties']['NAME_2']
        coords = feature['geometry']['coordinates']
        
        # Handle different geometry types
        if feature['geometry']['type'] == 'Polygon':
            all_coords = coords[0]
        elif feature['geometry']['type'] == 'MultiPolygon':
            all_coords = [coord for polygon in coords for coord in polygon[0]]
        else:
            continue
        
        # Calculate centroid
        lons = [coord[0] for coord in all_coords]
        lats = [coord[1] for coord in all_coords]
        centroids[district_name] = {
            'lon': np.mean(lons),
            'lat': np.mean(lats)
        }
    
    # Create choropleth map with unique colors
    fig = go.Figure()
    
    for _, row in district_counts.iterrows():
        district = row['district']
        client_count = row['client_count']
        color = row['color']
        
        # Find matching GeoJSON feature
        for feature in rwanda_geojson['features']:
            if feature['properties']['NAME_2'] == district:
                fig.add_trace(go.Choropleth(
                    geojson={"type": "FeatureCollection", "features": [feature]},
                    locations=[district],
                    z=[client_count],
                    featureidkey='properties.NAME_2',
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    hovertemplate=f'<b>{district}</b><br>Clients: {client_count}<extra></extra>',
                    marker_line_color='white',
                    marker_line_width=1.5
                ))
                break
    
    # Add district name and client count labels
    for district, centroid in centroids.items():
        if district in district_counts['district'].values:
            client_count = district_counts[district_counts['district'] == district]['client_count'].values[0]
            
            fig.add_trace(go.Scattergeo(
                lon=[centroid['lon']],
                lat=[centroid['lat']],
                text=f"<b>{district}</b><br>{client_count} clients",
                mode='text',
                textfont=dict(size=10, color='black', family='Arial Black'),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # Update map layout
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=False,
        showcoastlines=False,
        showland=False,
        showlakes=False
    )
    
    fig.update_layout(
        title=dict(
            text="<b>Rwanda Vehicle Clients Distribution by District</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2c3e50')
        ),
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        height=800,
        geo=dict(
            scope='africa',
            projection_type='mercator'
        )
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')


def get_district_summary_table(df):
    """
    Create a summary table of clients by district and province
    """
    summary = df.groupby(['province', 'district']).agg({
        'client_name': 'count',
        'estimated_income': 'mean',
        'selling_price': 'mean'
    }).reset_index()
    
    summary.columns = ['Province', 'District', 'Number of Clients', 'Avg Income', 'Avg Price']
    summary['Avg Income'] = summary['Avg Income'].round(2)
    summary['Avg Price'] = summary['Avg Price'].round(2)
    summary = summary.sort_values(['Province', 'District'])
    
    return summary.to_html(
        classes="table table-bordered table-striped table-sm",
        index=False,
        justify="center"
    )
