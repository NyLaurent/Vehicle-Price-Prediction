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
    Uses Mapbox choropleth with Reds color scale and static text labels for district names and client counts.
    """
    # Count clients per district
    district_counts = df['district'].value_counts().reset_index()
    district_counts.columns = ['district', 'client_count']
    district_counts['district'] = district_counts['district'].str.strip()
    
    # Load GeoJSON file
    geojson_path = os.path.join('dummy-data', 'rwanda_districts.geojson')
    with open(geojson_path, 'r', encoding='utf-8') as f:
        rwanda_geojson = json.load(f)
    
    # Calculate centroids for district labels
    centroids = []
    for feature in rwanda_geojson['features']:
        name = feature['properties']['NAME_2'].strip()
        feature['id'] = name
        coords = feature['geometry']['coordinates']
        
        # Extract all coordinates
        all_lons = []
        all_lats = []
        
        def extract_coords(c_list):
            for item in c_list:
                if isinstance(item[0], (int, float)):
                    all_lons.append(item[0])
                    all_lats.append(item[1])
                else:
                    extract_coords(item)
        
        extract_coords(coords)
        
        if all_lons and all_lats:
            centroids.append({
                'district': name,
                'lat': np.mean(all_lats),
                'lon': np.mean(all_lons)
            })
    
    centroid_df = pd.DataFrame(centroids)
    
    # Merge counts with centroids
    label_df = pd.merge(centroid_df, district_counts, on='district', how='left')
    label_df['client_count'] = label_df['client_count'].fillna(0).astype(int)
    
    # Formatted label text
    label_df['text'] = label_df['district'] + "<br>" + label_df['client_count'].astype(str)
    
    # Create base choropleth map with Mapbox
    fig = px.choropleth_mapbox(
        district_counts,
        geojson=rwanda_geojson,
        locations='district',
        color='client_count',
        color_continuous_scale="Blues",
        mapbox_style="carto-positron",
        center={"lat": -1.94, "lon": 30.06},
        zoom=7.8,
        opacity=0.6,
        title="<b>Rwanda Vehicle Clients Distribution by District</b>",
        labels={'client_count': 'Total Clients'}
    )
    
    # Add static text labels
    fig.add_trace(go.Scattermapbox(
        lat=label_df['lat'],
        lon=label_df['lon'],
        mode='text',
        text=label_df['text'],
        textfont={'size': 10, 'color': 'black', 'weight': 'bold'},
        hoverinfo='none',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        height=800,
        dragmode="zoom",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2c3e50')
        )
    )
    
    fig.update_mapboxes(
        center={"lat": -1.94, "lon": 30.06},
        zoom=7.8
    )
    
    fig.update_traces(
        marker_line_width=1,
        marker_line_color="darkblue",
        selector=dict(type='choroplethmapbox')
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn', config={'scrollZoom': True})


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
