import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import scipy.stats as stats

# Set page configuration
st.set_page_config(
    page_title="Amazon Reviews Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('https://www.dropbox.com/scl/fi/shs46nepo5mhxwxlmjpmc/combined_results.csv?rlkey=ipvo0uf5qaukry4h4ecdxe9rp&st=t50nov1n&dl=1')
    # Extract just the part after "Category_X " in the supplier column
    data['supplier'] = data['supplier'].str.split('Category_[0-9]+ ', regex=True).str[1]
    
    # Load original data with timestamps
    try:
        og_data = pd.read_csv("https://www.dropbox.com/scl/fi/riy1bbx5184n905ddbwkn/Amazon_Grocery_Data_02_16.csv?rlkey=sq6n0ktf9f0bqtvi3qmswfvxv&st=hxje4m5o&dl=1")
    except:
        og_data = pd.read_csv('https://www.dropbox.com/scl/fi/riy1bbx5184n905ddbwkn/Amazon_Grocery_Data_02_16.csv?rlkey=sq6n0ktf9f0bqtvi3qmswfvxv&st=hxje4m5o&dl=1')
    
    # Create review_id column
    og_data = og_data.reset_index()
    og_data['review_id'] = 'r' + og_data['index'].astype(str)
    og_data.drop('index', axis=1, inplace=True)
    
    # Add reviewTime from og_data to data based on review_id
    data = data.merge(og_data[['review_id', 'reviewTime']], on='review_id', how='left')
    
    # Convert reviewTime to datetime
    data['reviewTime'] = pd.to_datetime(data['reviewTime'])
    
    # Extract month and year for time series analysis
    data['review_month'] = data['reviewTime'].dt.to_period('M')
    
    return data

# Load the data
data = load_data()

# Calculate global averages for comparisons
global_negative_ratio = (data['sentiment'] == 'negative').mean()
global_avg_rating = data['overall_rating'].mean()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "Supplier Analysis", "Delivery Partner Analysis", 
     "Product Category Analysis", "Deep Dive", "Time Series Analysis"]
)

# Helper function to create radar charts
def create_radar_chart(df, category_column, aspect_column, sentiment_column, n_column, normalize=False, title=None):
    # Group by category and aspect, calculate average sentiment and count
    grouped = df.groupby([category_column, aspect_column]).agg({
        sentiment_column: 'mean',
        'review_id': 'count'
    }).reset_index()
    
    grouped.rename(columns={'review_id': n_column}, inplace=True)
    
    # Pivot to get aspects as columns
    pivot_df = grouped.pivot(index=category_column, columns=aspect_column, values=sentiment_column)
    
    # Get the categories with the most reviews
    top_categories = df[category_column].value_counts().nlargest(8).index.tolist()
    
    # Filter pivot_df to include only top categories
    pivot_df = pivot_df.loc[pivot_df.index.isin(top_categories)]
    
    # Get top aspects to ensure consistency across charts
    # This ensures all important aspects are included even if they're missing for some categories
    important_aspects = ['price', 'taste', 'quality', 'ingredients',
       'delivery', 'appearance', 'packaging']
    frequency_aspects = df['aspect'].value_counts().nlargest(10).index.tolist()
    
    # Combine important aspects with frequency-based aspects
    all_aspects = list(set(important_aspects + frequency_aspects))
    
    # Add missing columns with NaN values
    for aspect in all_aspects:
        if aspect not in pivot_df.columns:
            pivot_df[aspect] = np.nan
    
    # Filter to include only the aspects we want to display
    pivot_df = pivot_df[all_aspects]
    
    # Fill NaN values with the mean of the column to avoid gaps in the chart
    pivot_df = pivot_df.fillna(pivot_df.mean())
    
    # Normalize if requested - using min-max scaling instead of z-score
    if normalize:
        for col in pivot_df.columns:
            if not pivot_df[col].isna().all():  # Avoid columns with all NaN
                col_min = pivot_df[col].min()
                col_max = pivot_df[col].max()
                if col_max > col_min:  # Avoid division by zero
                    pivot_df[col] = (pivot_df[col] - col_min) / (col_max - col_min)
    
    # Create radar chart
    fig = go.Figure()
    
    for category in pivot_df.index:
        values = pivot_df.loc[category].tolist()
        # Add the first value again to close the loop
        values.append(values[0])
        
        aspects = pivot_df.columns.tolist()
        # Add the first aspect again to close the loop
        aspects.append(aspects[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=aspects,
            fill='toself',
            name=category
        ))
    
    # Set chart title
    if title is None:
        title = f"Sentiment Comparison by {category_column} and Aspect"
        if normalize:
            title = f"Min-Max Scaled {title}"
    
    # Set y-axis range based on normalization
    y_range = [0, 1]
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=y_range
            )
        ),
        showlegend=True,
        height=600,
        title=title
    )
    
    return fig

# Helper function to create bar charts
def create_sentiment_bar_chart(df, category_column, sentiment_filter=None):
    if sentiment_filter:
        df = df[df['sentiment'] == sentiment_filter]
    
    # Group by category and calculate negative sentiment rate and count
    grouped = df.groupby(category_column).agg({
        'sentiment': lambda x: (x == 'negative').mean(),
        'review_id': 'count'
    }).reset_index()
    
    grouped.columns = [category_column, 'negative_ratio', 'review_count']
    grouped = grouped.sort_values('negative_ratio', ascending=False)
    
    # Only include categories with at least 10 reviews
    grouped = grouped[grouped['review_count'] >= 10]
    
    # Take top 20 for better visualization
    grouped = grouped.head(20)
    
    # Create bar chart with review count as text
    fig = px.bar(
        grouped,
        x=category_column,
        y='negative_ratio',
        color='negative_ratio',
        text='review_count',
        color_continuous_scale='RdYlGn_r',  # Reversed scale: red for high negative
        title=f"Negative Sentiment Ratio by {category_column} (with review count)",
        height=500
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(yaxis_title='Negative Sentiment Ratio', xaxis_title=category_column)
    
    return fig

# Helper function for time series analysis
def create_time_series(df, category_column, selected_category=None):
    if selected_category:
        df = df[df[category_column] == selected_category]
    
    # Group by month and calculate sentiment ratios
    monthly = df.groupby('review_month').agg({
        'sentiment': lambda x: (x == 'negative').mean(),
        'review_id': 'count'
    }).reset_index()
    
    monthly.columns = ['month', 'negative_ratio', 'review_count']
    monthly['month'] = monthly['month'].astype(str)
    
    # Create the time series plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add negative sentiment ratio line
    fig.add_trace(
        go.Scatter(
            x=monthly['month'],
            y=monthly['negative_ratio'],
            name="Negative Sentiment Ratio",
            line=dict(color='red')
        ),
        secondary_y=False
    )
    
    # Add review count bars
    fig.add_trace(
        go.Bar(
            x=monthly['month'],
            y=monthly['review_count'],
            name="Review Count",
            marker=dict(color='lightblue'),
            opacity=0.7
        ),
        secondary_y=True
    )
    
    # Set titles
    title = f"Sentiment Trend Over Time" if not selected_category else f"Sentiment Trend for {selected_category}"
    fig.update_layout(
        title_text=title,
        height=500
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Negative Sentiment Ratio", secondary_y=False)
    fig.update_yaxes(title_text="Review Count", secondary_y=True)
    
    return fig

# Aspect breakdown helper function
def create_aspect_breakdown(df, category_column, selected_category):
    # Filter data for the selected category
    filtered_df = df[df[category_column] == selected_category]
    
    # Get other categories data for comparison
    others_df = df[df[category_column] != selected_category]
    
    # Group by aspect for the selected category
    aspect_data = filtered_df.groupby('aspect').agg({
        'sentiment': lambda x: (x == 'negative').mean(),
        'review_id': 'count'
    }).reset_index()
    
    aspect_data.columns = ['aspect', 'negative_ratio', 'review_count']
    aspect_data = aspect_data.sort_values('review_count', ascending=False)
    
    # Calculate average negative ratio by aspect for other categories
    others_avg = others_df.groupby('aspect').agg({
        'sentiment': lambda x: (x == 'negative').mean()
    }).reset_index()
    others_avg.columns = ['aspect', 'others_negative_ratio']
    
    # Merge with the main aspect data
    aspect_data = aspect_data.merge(others_avg, on='aspect', how='left')
    
    # Create a horizontal bar chart sorted by review count
    fig = px.bar(
        aspect_data,
        y='aspect',
        x='negative_ratio',
        orientation='h',
        color='negative_ratio',
        text='review_count',
        color_continuous_scale='RdYlGn_r',  # Reversed scale: red for high negative
        title=f"Aspect Breakdown for {selected_category} (with review count)",
        height=500
    )
    
    # Add markers for the average of other categories
    fig.add_trace(go.Scatter(
        x=aspect_data['others_negative_ratio'],
        y=aspect_data['aspect'],
        mode='markers',
        marker=dict(
            symbol='x',
            size=10,
            color='black',
            line=dict(width=2)
        ),
        name='Avg. of Other Categories'
    ))
    
    fig.update_traces(texttemplate='%{text}', textposition='outside', selector=dict(type='bar'))
    fig.update_layout(xaxis_title='Negative Sentiment Ratio', yaxis_title='Aspect')
    
    return fig

# Add comparison to average function for deep dive
def create_comparison_to_avg(df, category_column, selected_category):
    # Filter data for the selected category
    filtered_df = df[df[category_column] == selected_category]
    
    # Calculate metrics for the selected category
    cat_negative_ratio = (filtered_df['sentiment'] == 'negative').mean()
    
    # Calculate metrics for all other categories
    others_df = df[df[category_column] != selected_category]
    others_negative_ratio = (others_df['sentiment'] == 'negative').mean()
    
    # Create comparison dataframe
    comparison_data = pd.DataFrame({
        'Category': [selected_category, 'All Others', 'Global Average'],
        'Negative Sentiment Ratio': [cat_negative_ratio, others_negative_ratio, global_negative_ratio]
    })
    
    # Create bar chart
    fig = px.bar(
        comparison_data,
        x='Category',
        y='Negative Sentiment Ratio',
        color='Negative Sentiment Ratio',
        color_continuous_scale='RdYlGn_r',  # Reversed scale: red for high negative
        title=f"Negative Sentiment Comparison: {selected_category} vs Average",
        height=400
    )
    
    fig.update_layout(yaxis_title='Negative Sentiment Ratio')
    
    return fig

# Add aspect-specific time series function
def create_aspect_time_series(df, category_column, selected_category, aspect):
    # Filter data for the selected category and aspect
    category_data = df[(df[category_column] == selected_category) & (df['aspect'] == aspect)]
    
    # Filter global data for the specific aspect (for comparison)
    global_aspect_data = df[df['aspect'] == aspect]
    
    # Group category data by month
    if len(category_data) > 0:
        category_monthly = category_data.groupby('review_month').agg({
            'sentiment': lambda x: (x == 'negative').mean(),
            'review_id': 'count'
        }).reset_index()
        
        category_monthly.columns = ['month', 'negative_ratio', 'review_count']
        category_monthly['month'] = category_monthly['month'].astype(str)
    else:
        # Create empty dataframe if no data
        category_monthly = pd.DataFrame(columns=['month', 'negative_ratio', 'review_count'])
    
    # Group global data by month
    global_monthly = global_aspect_data.groupby('review_month').agg({
        'sentiment': lambda x: (x == 'negative').mean(),
        'review_id': 'count'
    }).reset_index()
    
    global_monthly.columns = ['month', 'negative_ratio', 'review_count']
    global_monthly['month'] = global_monthly['month'].astype(str)
    
    # Create the time series plot with dual y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add selected category line
    if len(category_monthly) > 0:
        fig.add_trace(
            go.Scatter(
                x=category_monthly['month'],
                y=category_monthly['negative_ratio'],
                name=f"{selected_category}",
                line=dict(color='red', width=2)
            ),
            secondary_y=False
        )
        
        # Add review count bars
        fig.add_trace(
            go.Bar(
                x=category_monthly['month'],
                y=category_monthly['review_count'],
                name="Review Count",
                marker=dict(color='lightblue'),
                opacity=0.7
            ),
            secondary_y=True
        )
    
    # Add global average line
    fig.add_trace(
        go.Scatter(
            x=global_monthly['month'],
            y=global_monthly['negative_ratio'],
            name="Global Average",
            line=dict(color='black', width=2, dash='dash')
        ),
        secondary_y=False
    )
    
    # Set titles
    title = f"Negative Sentiment Trend for '{aspect}' Aspect - {selected_category} vs. Global Average"
    fig.update_layout(
        title_text=title,
        height=400,
        xaxis_title="Month"
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Negative Sentiment Ratio", secondary_y=False)
    fig.update_yaxes(title_text="Review Count", secondary_y=True)
    
    return fig

# Add function to get sample reviews with highest negative confidence for each aspect
def get_top_negative_reviews(df, category_column, selected_category, aspect, num_samples=5):
    # Filter data for the selected category and aspect
    filtered_df = df[(df[category_column] == selected_category) & (df['aspect'] == aspect)]
    
    # Sort by negative confidence (highest first)
    if 'negative_conf' in filtered_df.columns:
        top_reviews = filtered_df.sort_values('negative_conf', ascending=False).head(num_samples)
    else:
        # If negative_conf not available, use sentiment as fallback
        top_reviews = filtered_df[filtered_df['sentiment'] == 'negative'].head(num_samples)
    
    return top_reviews

# Pages
if page == "Overview":
    st.title("Amazon Reviews Sentiment Analysis Dashboard")
    
    # Dashboard description
    st.markdown("""
    This dashboard provides interactive visualizations of sentiment analysis from Amazon product reviews.
    Use the sidebar navigation to explore different aspects of the data.
    """)
    
    # Show dataset information and sentiment distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Information")
        st.write(f"Total reviews: {len(data):,}")
        st.write(f"Unique suppliers: {data['supplier'].nunique():,}")
        st.write(f"Unique delivery partners: {data['delivery_partner'].nunique():,}")
        st.write(f"Unique product categories: {data['product_category'].nunique():,}")
        st.write(f"Date range: {data['reviewTime'].min().date()} to {data['reviewTime'].max().date()}")
        
        # Dataset preview
        st.subheader("Data Preview")
        st.dataframe(data.head(5))
    
    with col2:
        # Overall sentiment distribution
        st.subheader("Overall Sentiment Distribution")
        sentiment_counts = data['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig = px.pie(
            sentiment_counts,
            values='Count',
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'},
            title="Distribution of Sentiments"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Get aspects for radar charts - including important aspects and top frequent aspects
    important_aspects = ['taste', 'quality', 'packaging', 'price', 'delivery', 'ingredients']
    frequency_aspects = data['aspect'].value_counts().nlargest(10).index.tolist()
    
    # Combine important aspects with frequency-based aspects
    all_aspects = list(set(important_aspects + frequency_aspects))
    filtered_data = data[data['aspect'].isin(all_aspects)]
    
    # Create sentiment score column for radar charts
    sentiment_numeric = filtered_data['sentiment'].map({'positive': 0, 'neutral': 0.5, 'negative': 1})
    radar_data = filtered_data.copy()
    radar_data['sentiment_score'] = sentiment_numeric
    
    # Supplier Aspect Comparison
    st.header("Key Dimension Comparisons")
    
    # 1. Supplier Aspect Comparison
    st.subheader("Supplier Aspect Comparison")
    
    # Create regular and normalized radar charts side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Regular radar chart for suppliers
        radar_chart = create_radar_chart(
            radar_data, 
            'supplier', 
            'aspect', 
            'sentiment_score', 
            'review_count',
            title="Negative Sentiment by Supplier and Aspect"
        )
        st.plotly_chart(radar_chart, use_container_width=True)
    
    with col2:
        # Normalized radar chart for suppliers
        normalized_radar = create_radar_chart(
            radar_data, 
            'supplier', 
            'aspect', 
            'sentiment_score', 
            'review_count',
            normalize=True,
            title="Min-Max Scaled Supplier Sentiment"
        )
        st.plotly_chart(normalized_radar, use_container_width=True)
    
    # 2. Delivery Partner Aspect Comparison
    st.subheader("Delivery Partner Aspect Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regular radar chart for delivery partners
        delivery_radar = create_radar_chart(
            radar_data, 
            'delivery_partner', 
            'aspect', 
            'sentiment_score', 
            'review_count',
            title="Negative Sentiment by Delivery Partner and Aspect"
        )
        st.plotly_chart(delivery_radar, use_container_width=True)
    
    with col2:
        # Normalized radar chart for delivery partners
        normalized_delivery_radar = create_radar_chart(
            radar_data, 
            'delivery_partner', 
            'aspect', 
            'sentiment_score', 
            'review_count',
            normalize=True,
            title="Min-Max Scaled Delivery Partner Sentiment"
        )
        st.plotly_chart(normalized_delivery_radar, use_container_width=True)
    
    # 3. Product Category Aspect Comparison
    st.subheader("Product Category Aspect Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regular radar chart for product categories
        category_radar = create_radar_chart(
            radar_data, 
            'product_category', 
            'aspect', 
            'sentiment_score', 
            'review_count',
            title="Negative Sentiment by Product Category and Aspect"
        )
        st.plotly_chart(category_radar, use_container_width=True)
    
    with col2:
        # Normalized radar chart for product categories
        normalized_category_radar = create_radar_chart(
            radar_data, 
            'product_category', 
            'aspect', 
            'sentiment_score', 
            'review_count',
            normalize=True,
            title="Min-Max Scaled Product Category Sentiment"
        )
        st.plotly_chart(normalized_category_radar, use_container_width=True)
    
    st.info("The min-max scaled charts highlight the relative differences between entities across aspects, making it easier to identify outliers.")

elif page == "Supplier Analysis":
    st.title("Supplier Analysis")
    
    # Suppliers with most reviews
    st.subheader("Most Reviewed Suppliers")
    supplier_counts = data['supplier'].value_counts().reset_index()
    supplier_counts.columns = ['Supplier', 'Review Count']
    supplier_counts = supplier_counts.head(15)
    fig = px.bar(
        supplier_counts,
        x='Supplier',
        y='Review Count',
        color='Review Count',
        title="Top 15 Suppliers by Review Count"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Supplier sentiment comparison
    st.subheader("Supplier Sentiment Comparison")
    supplier_sentiment_fig = create_sentiment_bar_chart(data, 'supplier')
    st.plotly_chart(supplier_sentiment_fig, use_container_width=True)
    
    # Radar chart for supplier aspect comparison
    st.subheader("Supplier Aspect Comparison")
    st.write("This radar chart shows how different suppliers compare across various aspects based on negative sentiment scores.")
    
    # Get top aspects by frequency
    top_aspects = data['aspect'].value_counts().nlargest(6).index.tolist()
    filtered_data = data[data['aspect'].isin(top_aspects)]
    
    # Create supplier aspect radar chart
    sentiment_numeric = filtered_data['sentiment'].map({'positive': 0, 'neutral': 0.5, 'negative': 1})
    radar_data = filtered_data.copy()
    radar_data['sentiment_score'] = sentiment_numeric
    
    radar_chart = create_radar_chart(radar_data, 'supplier', 'aspect', 'sentiment_score', 'review_count')
    st.plotly_chart(radar_chart, use_container_width=True)

elif page == "Delivery Partner Analysis":
    st.title("Delivery Partner Analysis")
    
    # Most active delivery partners
    st.subheader("Most Active Delivery Partners")
    delivery_counts = data['delivery_partner'].value_counts().reset_index()
    delivery_counts.columns = ['Delivery Partner', 'Review Count']
    fig = px.bar(
        delivery_counts,
        x='Delivery Partner',
        y='Review Count',
        color='Review Count',
        title="Delivery Partners by Review Count"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Delivery partner sentiment comparison
    st.subheader("Delivery Partner Sentiment Comparison")
    delivery_sentiment_fig = create_sentiment_bar_chart(data, 'delivery_partner')
    st.plotly_chart(delivery_sentiment_fig, use_container_width=True)
    
    # Radar chart for delivery partner aspect comparison
    st.subheader("Delivery Partner Aspect Comparison")
    st.write("This radar chart shows how different delivery partners compare across various aspects based on negative sentiment scores.")
    
    # Create delivery partner aspect radar chart
    sentiment_numeric = data['sentiment'].map({'positive': 0, 'neutral': 0.5, 'negative': 1})
    radar_data = data.copy()
    radar_data['sentiment_score'] = sentiment_numeric
    
    delivery_radar = create_radar_chart(radar_data, 'delivery_partner', 'aspect', 'sentiment_score', 'review_count')
    st.plotly_chart(delivery_radar, use_container_width=True)

elif page == "Product Category Analysis":
    st.title("Product Category Analysis")
    
    # Most reviewed product categories
    st.subheader("Most Reviewed Product Categories")
    category_counts = data['product_category'].value_counts().reset_index()
    category_counts.columns = ['Product Category', 'Review Count']
    fig = px.bar(
        category_counts,
        x='Product Category',
        y='Review Count',
        color='Review Count',
        title="Product Categories by Review Count"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Product category sentiment comparison
    st.subheader("Product Category Sentiment Comparison")
    category_sentiment_fig = create_sentiment_bar_chart(data, 'product_category')
    st.plotly_chart(category_sentiment_fig, use_container_width=True)
    
    # Radar chart for product category aspect comparison
    st.subheader("Product Category Aspect Comparison")
    st.write("This radar chart shows how different product categories compare across various aspects based on negative sentiment scores.")
    
    # Create product category aspect radar chart
    sentiment_numeric = data['sentiment'].map({'positive': 0, 'neutral': 0.5, 'negative': 1})
    radar_data = data.copy()
    radar_data['sentiment_score'] = sentiment_numeric
    
    category_radar = create_radar_chart(radar_data, 'product_category', 'aspect', 'sentiment_score', 'review_count')
    st.plotly_chart(category_radar, use_container_width=True)

elif page == "Deep Dive":
    st.title("Deep Dive Analysis")
    st.write("Select a dimension and specific value to dive deeper into the data.")
    
    # Selection widgets
    dimension = st.selectbox(
        "Select dimension to analyze:",
        ["supplier", "delivery_partner", "product_category"]
    )
    
    dimension_label = {
        "supplier": "Supplier",
        "delivery_partner": "Delivery Partner",
        "product_category": "Product Category"
    }
    
    # Get values for the selected dimension
    dimension_values = data[dimension].unique().tolist()
    selected_value = st.selectbox(f"Select {dimension_label[dimension]}:", dimension_values)
    
    # Filter data for the selected dimension and value
    filtered_data = data[data[dimension] == selected_value]
    
    # Display basic stats with comparison to average
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", f"{len(filtered_data):,}")
    with col2:
        # Calculate negative sentiment and delta from global average
        negative_ratio = (filtered_data['sentiment'] == 'negative').mean()
        delta = negative_ratio - global_negative_ratio
        st.metric("Negative Sentiment Ratio", f"{negative_ratio:.2%}", 
                 delta=f"{delta:.2%}", delta_color="inverse")
    with col3:
        # Calculate average rating and delta from global average
        avg_rating = filtered_data['overall_rating'].mean()
        rating_delta = avg_rating - global_avg_rating
        st.metric("Average Rating", f"{avg_rating:.2f}/5", 
                 delta=f"{rating_delta:.2f}", delta_color="normal")
    
    # Aspect breakdown
    st.subheader(f"Aspect Breakdown for {selected_value}")
    aspect_fig = create_aspect_breakdown(data, dimension, selected_value)
    st.plotly_chart(aspect_fig, use_container_width=True)
    
    # Sample negative reviews for each aspect
    st.subheader(f"Sample Negative Reviews for {selected_value}")
    
    # Get the top aspects for this dimension value
    aspect_counts = filtered_data['aspect'].value_counts().reset_index()
    aspect_counts.columns = ['aspect', 'count']
    top_aspects = aspect_counts.head(5)['aspect'].tolist()
    
    if len(top_aspects) > 0:
        # Create tabs for each aspect
        aspect_tabs = st.tabs(top_aspects)
        
        # For each tab/aspect, show sample reviews
        for i, aspect in enumerate(top_aspects):
            with aspect_tabs[i]:
                st.write(f"### Top 5 Most Negative Reviews for '{aspect}' Aspect")
                
                top_reviews = get_top_negative_reviews(data, dimension, selected_value, aspect)
                
                if len(top_reviews) > 0:
                    for _, review in top_reviews.iterrows():
                        with st.expander(f"Review ID: {review['review_id']} | Negative Conf: {review['negative_conf']:.3f} | Rating: {review['overall_rating']}/5"):
                            st.markdown(f"**Review Text:** {review['review_text']}")
                            st.markdown(f"**Entity:** {review['entity']}")
                            st.markdown(f"**Sentiment:** {review['sentiment']}")
                            if 'verified' in review and review['verified']:
                                st.markdown("✓ **Verified Purchase**")
                else:
                    st.write("No negative reviews available for this aspect.")
    else:
        st.write("No aspect data available for sample reviews.")
    
    # Determine the other dimensions to analyze
    other_dimensions = [d for d in ["supplier", "delivery_partner", "product_category"] if d != dimension]
    
    # For each other dimension, show a breakdown
    for other_dimension in other_dimensions:
        other_label = dimension_label[other_dimension]
        st.subheader(f"{other_label} Breakdown for {selected_value}")
        
        # Group by the other dimension
        other_data = filtered_data.groupby(other_dimension).agg({
            'sentiment': lambda x: (x == 'negative').mean(),
            'review_id': 'count'
        }).reset_index()
        
        other_data.columns = [other_dimension, 'negative_ratio', 'review_count']
        other_data = other_data.sort_values('review_count', ascending=False).head(10)
        
        # Create bar chart
        fig = px.bar(
            other_data,
            x=other_dimension,
            y='negative_ratio',
            color='negative_ratio',
            text='review_count',
            color_continuous_scale='RdYlGn_r',  # Reversed scale: red for high negative
            title=f"Top 10 {other_label}s for {selected_value} by Review Count",
            height=400
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(yaxis_title='Negative Sentiment Ratio', xaxis_title=other_label)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # New aspect-specific time series analysis
    st.subheader(f"Aspect-Specific Sentiment Trends for {selected_value}")
    
    # Get the top aspects for this dimension value
    aspect_counts = filtered_data['aspect'].value_counts().reset_index()
    aspect_counts.columns = ['aspect', 'count']
    top_aspects = aspect_counts.head(5)['aspect'].tolist()
    
    if len(top_aspects) > 0:
        st.write("These charts show how negative sentiment for each aspect has changed over time, compared to the global average for that aspect.")
        
        # Create time series for each top aspect
        for aspect in top_aspects:
            aspect_time_fig = create_aspect_time_series(data, dimension, selected_value, aspect)
            st.plotly_chart(aspect_time_fig, use_container_width=True)
    else:
        st.write("No aspect data available for time series analysis.")

elif page == "Time Series Analysis":
    st.title("Time Series Analysis")
    
    # Overall time series
    st.subheader("Overall Sentiment Trend Over Time")
    overall_time_fig = create_time_series(data, None)
    st.plotly_chart(overall_time_fig, use_container_width=True)
    
    # Dimension selection for time series
    dimension = st.selectbox(
        "Select dimension for time series analysis:",
        ["supplier", "delivery_partner", "product_category"]
    )
    
    dimension_label = {
        "supplier": "Supplier",
        "delivery_partner": "Delivery Partner",
        "product_category": "Product Category"
    }
    
    # Get top values for the selected dimension by review count
    top_values = data[dimension].value_counts().nlargest(5).index.tolist()
    
    # Create time series for each top value
    st.subheader(f"Sentiment Trends for Top 5 {dimension_label[dimension]}s")
    
    for value in top_values:
        st.write(f"### {value}")
        value_time_fig = create_time_series(data, dimension, value)
        st.plotly_chart(value_time_fig, use_container_width=True)
    
    # Heatmap of monthly sentiment by dimension
    st.subheader(f"Monthly Sentiment Heatmap by {dimension_label[dimension]}")
    
    # Group data by month and dimension
    monthly_dimension = data.groupby(['review_month', dimension]).agg({
        'sentiment': lambda x: (x == 'negative').mean(),
        'review_id': 'count'
    }).reset_index()
    
    monthly_dimension.columns = ['month', dimension, 'negative_ratio', 'review_count']
    monthly_dimension['month'] = monthly_dimension['month'].astype(str)
    
    # Filter to include only values with sufficient reviews
    value_counts = monthly_dimension.groupby(dimension)['review_count'].sum()
    top_values = value_counts.nlargest(10).index.tolist()
    heatmap_data = monthly_dimension[monthly_dimension[dimension].isin(top_values)]
    
    # Create pivot table for heatmap
    heatmap_pivot = heatmap_data.pivot(index='month', columns=dimension, values='negative_ratio')
    
    # Create heatmap
    fig = px.imshow(
        heatmap_pivot,
        color_continuous_scale='RdYlGn_r',  # Reversed scale: red for high negative
        title=f"Monthly Negative Sentiment Ratio by {dimension_label[dimension]}",
        labels=dict(x=dimension_label[dimension], y="Month", color="Negative Ratio"),
        zmin=0,  # Set minimum value for color scale
        zmax=0.4  # Set maximum value for color scale
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # NEW SECTION: Aspect-specific heatmaps
    st.header("Aspect-Specific Sentiment Heatmaps")
    st.write("These heatmaps show how negative sentiment for specific aspects changes over time across different entities.")
    
    # Get most common aspects
    important_aspects = ['taste', 'quality', 'packaging', 'price', 'delivery', 'ingredients']
    frequency_aspects = data['aspect'].value_counts().nlargest(10).index.tolist()
    all_aspects = list(set(important_aspects + frequency_aspects))
    
    # Create tabs for each aspect
    aspect_tabs = st.tabs(all_aspects)
    
    for i, aspect in enumerate(all_aspects):
        with aspect_tabs[i]:
            st.write(f"### Negative Sentiment Heatmap for '{aspect}' Aspect")
            
            # Filter data for this aspect
            aspect_data = data[data['aspect'] == aspect]
            
            if len(aspect_data) > 0:
                # Group data by month and dimension
                monthly_aspect = aspect_data.groupby(['review_month', dimension]).agg({
                    'sentiment': lambda x: (x == 'negative').mean(),
                    'review_id': 'count'
                }).reset_index()
                
                monthly_aspect.columns = ['month', dimension, 'negative_ratio', 'review_count']
                monthly_aspect['month'] = monthly_aspect['month'].astype(str)
                
                # Filter to include only values with sufficient reviews
                value_counts = monthly_aspect.groupby(dimension)['review_count'].sum()
                if len(value_counts) > 0:
                    top_values = value_counts.nlargest(10).index.tolist()
                    aspect_heatmap_data = monthly_aspect[monthly_aspect[dimension].isin(top_values)]
                    
                    # Create pivot table for heatmap
                    aspect_heatmap_pivot = aspect_heatmap_data.pivot(index='month', columns=dimension, values='negative_ratio')
                    
                    if not aspect_heatmap_pivot.empty:
                        # Create heatmap
                        fig = px.imshow(
                            aspect_heatmap_pivot,
                            color_continuous_scale='RdYlGn_r',  # Reversed scale: red for high negative
                            title=f"Monthly Negative Sentiment for '{aspect}' by {dimension_label[dimension]}",
                            labels=dict(x=dimension_label[dimension], y="Month", color="Negative Ratio"),
                            zmin=0,  # Set minimum value for color scale
                            zmax=0.4  # Set maximum value for color scale
                        )
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show review count heatmap for context
                        review_count_pivot = aspect_heatmap_data.pivot(index='month', columns=dimension, values='review_count')
                        
                        fig_count = px.imshow(
                            review_count_pivot,
                            color_continuous_scale='Blues',
                            title=f"Monthly Review Count for '{aspect}' by {dimension_label[dimension]}",
                            labels=dict(x=dimension_label[dimension], y="Month", color="Review Count")
                        )
                        
                        fig_count.update_layout(height=500)
                        st.plotly_chart(fig_count, use_container_width=True)
                    else:
                        st.write(f"Not enough data to create heatmap for '{aspect}' aspect.")
                else:
                    st.write(f"Not enough data to create heatmap for '{aspect}' aspect.")
            else:
                st.write(f"No data available for '{aspect}' aspect.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Amazon Reviews Sentiment Analysis Dashboard") 