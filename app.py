import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        border-color: #FF4B4B;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #FF4B4B;
        text-align: center;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #4B8BFF;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Feature descriptions
FEATURE_INFO = {
    'MedInc': {
        'name': 'Median Income',
        'description': 'Median income in block group (in tens of thousands of dollars)',
        'min': 0.5, 'max': 15.0, 'default': 3.5, 'step': 0.1
    },
    'HouseAge': {
        'name': 'House Age',
        'description': 'Median house age in block group (in years)',
        'min': 1.0, 'max': 52.0, 'default': 25.0, 'step': 1.0
    },
    'AveRooms': {
        'name': 'Average Rooms',
        'description': 'Average number of rooms per household',
        'min': 1.0, 'max': 10.0, 'default': 5.0, 'step': 0.1
    },
    'AveBedrms': {
        'name': 'Average Bedrooms',
        'description': 'Average number of bedrooms per household',
        'min': 0.5, 'max': 5.0, 'default': 1.0, 'step': 0.1
    },
    'Population': {
        'name': 'Population',
        'description': 'Block group population',
        'min': 100.0, 'max': 5000.0, 'default': 1500.0, 'step': 50.0
    },
    'AveOccup': {
        'name': 'Average Occupancy',
        'description': 'Average number of household members',
        'min': 1.0, 'max': 6.0, 'default': 3.0, 'step': 0.1
    },
    'Latitude': {
        'name': 'Latitude',
        'description': 'Block group latitude coordinate',
        'min': 32.5, 'max': 42.0, 'default': 34.0, 'step': 0.1
    },
    'Longitude': {
        'name': 'Longitude',
        'description': 'Block group longitude coordinate',
        'min': -124.5, 'max': -114.0, 'default': -118.0, 'step': 0.1
    }
}

def main():
    # Header
    st.title("🏠 House Price Prediction App")
    st.markdown("### Predict California housing prices using XGBoost")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Model Information")
        st.markdown("""
        <div class="info-box">
        <b>Model:</b> XGBoost Regressor<br>
        <b>Dataset:</b> California Housing<br>
        <b>Features:</b> 8 input variables<br>
        <b>Target:</b> House price (in $100,000s)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("📊 Input Method")
        input_method = st.radio(
            "Choose how to input data:",
            ["Manual Input", "Batch Prediction (CSV)"],
            help="Select whether to predict for a single house or multiple houses"
        )
    
    # Load model
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'house_price_model.pkl' is in the same directory.")
        return
    
    if input_method == "Manual Input":
        manual_prediction(model)
    else:
        batch_prediction(model)

def manual_prediction(model):
    st.header("📝 Enter House Features")
    
    col1, col2 = st.columns(2)
    
    features = {}
    
    with col1:
        st.subheader("Economic & Property Features")
        for feature in ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']:
            info = FEATURE_INFO[feature]
            features[feature] = st.number_input(
                info['name'],
                min_value=info['min'],
                max_value=info['max'],
                value=info['default'],
                step=info['step'],
                help=info['description']
            )
    
    with col2:
        st.subheader("Location & Demographics")
        for feature in ['Population', 'AveOccup', 'Latitude', 'Longitude']:
            info = FEATURE_INFO[feature]
            features[feature] = st.number_input(
                info['name'],
                min_value=info['min'],
                max_value=info['max'],
                value=info['default'],
                step=info['step'],
                help=info['description']
            )
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict House Price", type="primary")
    
    if predict_button:
        # Prepare input data
        input_data = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Convert to actual price (prediction is in $100,000s)
        price = prediction * 100000
        
        # Display prediction
        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")
        
        col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
        with col_result2:
            st.markdown(f"""
            <div class="prediction-box">
                <h1 style="color: #FF4B4B; margin: 0;">${price:,.2f}</h1>
                <p style="color: #666; margin-top: 0.5rem;">Estimated House Price</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance visualization
        st.markdown("### 📊 Input Feature Values")
        
        # Create a bar chart of input features
        feature_df = pd.DataFrame({
            'Feature': [FEATURE_INFO[k]['name'] for k in features.keys()],
            'Value': list(features.values())
        })
        
        fig = px.bar(
            feature_df,
            x='Feature',
            y='Value',
            color='Value',
            color_continuous_scale='Blues',
            title='Your Input Feature Values'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display input summary
        with st.expander("📋 View Input Summary"):
            summary_df = pd.DataFrame({
                'Feature': [FEATURE_INFO[k]['name'] for k in features.keys()],
                'Value': [f"{v:.2f}" for v in features.values()],
                'Description': [FEATURE_INFO[k]['description'] for k in features.keys()]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

def batch_prediction(model):
    st.header("📂 Batch Prediction from CSV")
    
    st.markdown("""
    <div class="info-box">
    Upload a CSV file with the following columns:<br>
    <b>MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude</b>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample data download
    sample_data = pd.DataFrame({
        'MedInc': [3.5, 5.0, 2.8],
        'HouseAge': [25.0, 15.0, 40.0],
        'AveRooms': [5.0, 6.2, 4.5],
        'AveBedrms': [1.0, 1.1, 0.9],
        'Population': [1500.0, 2000.0, 1200.0],
        'AveOccup': [3.0, 2.8, 3.2],
        'Latitude': [34.0, 37.5, 33.8],
        'Longitude': [-118.0, -122.0, -117.5]
    })
    
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Sample CSV Template",
        data=csv,
        file_name="sample_house_data.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                           'Population', 'AveOccup', 'Latitude', 'Longitude']
            
            if not all(col in df.columns for col in required_cols):
                st.error(f"❌ CSV must contain columns: {', '.join(required_cols)}")
                return
            
            # Display uploaded data
            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.info(f"Total rows: {len(df)}")
            
            # Predict button
            if st.button("🔮 Predict All Prices", type="primary"):
                # Make predictions
                predictions = model.predict(df[required_cols])
                
                # Add predictions to dataframe
                df['Predicted_Price_100k'] = predictions
                df['Predicted_Price_USD'] = predictions * 100000
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Properties", len(df))
                with col2:
                    st.metric("Average Price", f"${df['Predicted_Price_USD'].mean():,.2f}")
                with col3:
                    st.metric("Minimum Price", f"${df['Predicted_Price_USD'].min():,.2f}")
                with col4:
                    st.metric("Maximum Price", f"${df['Predicted_Price_USD'].max():,.2f}")
                
                # Results table
                st.dataframe(
                    df.style.format({'Predicted_Price_USD': '${:,.2f}', 'Predicted_Price_100k': '{:.2f}'}),
                    use_container_width=True
                )
                
                # Visualization
                st.subheader("📊 Price Distribution")
                fig = px.histogram(
                    df,
                    x='Predicted_Price_USD',
                    nbins=30,
                    title='Distribution of Predicted House Prices',
                    labels={'Predicted_Price_USD': 'Predicted Price (USD)'},
                    color_discrete_sequence=['#FF4B4B']
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                result_csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Predictions CSV",
                    data=result_csv,
                    file_name="house_price_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

if __name__ == "__main__":
    main()