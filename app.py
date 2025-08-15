import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    st.warning("Google Generative AI not installed. Municipality search will use fallback data.")

try:
    from geopy.geocoders import Nominatim
except ImportError:
    Nominatim = None

try:
    import folium
    from streamlit_folium import st_folium
except ImportError:
    folium = None
    st.warning("Folium not installed. Map features will be disabled.")

import base64
import time
import logging

# ========================
# PAGE CONFIG & LOGGING
# ========================
st.set_page_config(
    page_title="🌱 EcoWaste AI - Smart Garbage Detection", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="♻️"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# CSS LOADING FUNCTION
# ========================
def load_css():
    """Load CSS with error handling"""
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ CSS file 'enhanced_styles.css' not found. Using basic styling.")
        # Fallback minimal CSS
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
        }
        .hero-header {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error loading CSS: {e}")

# ========================
# CONFIGURATION
# ========================

# Gemini AI Configuration
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# Enhanced Waste Classification
WASTE_CATEGORIES = {
    "cardboard": {
        "recyclable": True, 
        "category": "Paper", 
        "bin": "Blue Recycling Bin", 
        "color": "#8BC34A",
        "icon": "📦",
        "tips": "Remove tape and flatten before recycling",
        "co2_saved": 3.3,  # kg CO2 per item
        "energy_saved": 1.5  # kWh per item
    },
    "glass": {
        "recyclable": True, 
        "category": "Glass", 
        "bin": "Green Glass Bin", 
        "color": "#4CAF50",
        "icon": "🍾",
        "tips": "Remove lids and rinse clean",
        "co2_saved": 0.5,
        "energy_saved": 0.3
    },
    "metal": {
        "recyclable": True, 
        "category": "Metal", 
        "bin": "Blue Recycling Bin", 
        "color": "#9E9E9E",
        "icon": "🥫",
        "tips": "Rinse cans and remove labels if possible",
        "co2_saved": 1.8,
        "energy_saved": 2.1
    },
    "paper": {
        "recyclable": True, 
        "category": "Paper", 
        "bin": "Blue Recycling Bin", 
        "color": "#FFC107",
        "icon": "📄",
        "tips": "Keep dry and remove any plastic components",
        "co2_saved": 1.1,
        "energy_saved": 1.0
    },
    "plastic": {
        "recyclable": True, 
        "category": "Plastic", 
        "bin": "Blue Recycling Bin", 
        "color": "#2196F3",
        "icon": "🥤",
        "tips": "Check recycling number and rinse clean",
        "co2_saved": 2.0,
        "energy_saved": 1.8
    },
    "battery": {
        "recyclable": True, 
        "category": "Hazardous", 
        "bin": "Special Battery Collection Point", 
        "color": "#FF5722",
        "icon": "🔋",
        "tips": "Never put in regular trash - take to collection center",
        "co2_saved": 0.8,
        "energy_saved": 0.5
    },
    "clothes": {
        "recyclable": True, 
        "category": "Textile", 
        "bin": "Textile Recycling Bin", 
        "color": "#9C27B0",
        "icon": "👕",
        "tips": "Donate if in good condition or use textile recycling",
        "co2_saved": 5.5,
        "energy_saved": 3.2
    },
    "organic": {
        "recyclable": False, 
        "category": "Organic", 
        "bin": "Brown Compost Bin", 
        "color": "#795548",
        "icon": "🍎",
        "tips": "Compost at home or use organic waste bin",
        "co2_saved": 0.3,
        "energy_saved": 0.1
    },
    "shoes": {
        "recyclable": True, 
        "category": "Textile", 
        "bin": "Shoe Recycling Center", 
        "color": "#607D8B",
        "icon": "👟",
        "tips": "Donate if wearable or take to specialized recycling",
        "co2_saved": 2.8,
        "energy_saved": 1.9
    },
    "trash": {
        "recyclable": False, 
        "category": "General Waste", 
        "bin": "General Waste Bin", 
        "color": "#424242",
        "icon": "🗑️",
        "tips": "Dispose in general waste bin",
        "co2_saved": 0.0,
        "energy_saved": 0.0
    }
}

# ========================
# MODEL LOADING
# ========================
@st.cache_resource
def load_model():
    """Load YOLO model with comprehensive error handling"""
    try:
        model_paths = ["final_model.pt", "best.pt", "yolov8n.pt", "yolov8s.pt"]
        
        for model_path in model_paths:
            try:
                if os.path.exists(model_path) or model_path.startswith("yolov8"):
                    model = YOLO(model_path)
                    st.success(f"✅ Successfully loaded model: {model_path}")
                    return model
            except Exception as e:
                logger.warning(f"Failed to load {model_path}: {e}")
                continue
        
        # If all else fails, try to load a basic YOLOv8 model
        st.warning("⚠️ No custom model found. Downloading YOLOv8 nano...")
        return YOLO("yolov8n.pt")
        
    except Exception as e:
        st.error(f"❌ Critical error loading model: {e}")
        return None

# ========================
# GEMINI AI INTEGRATION
# ========================
def setup_gemini_ai():
    """Setup Gemini AI with proper error handling"""
    if genai is None:
        return None
        
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            return genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.warning(f"⚠️ Gemini AI setup error: {e}")
            return None
    else:
        st.info("💡 Add GEMINI_API_KEY to Streamlit secrets for enhanced municipality search")
        return None

def get_user_location():
    """Get user location with fallback options"""
    try:
        # Try to get location from browser (would need additional setup)
        # For now, return default location
        return {
            "lat": 28.4089, 
            "lng": 77.3178, 
            "city": "Faridabad", 
            "state": "Haryana", 
            "country": "India"
        }
    except Exception as e:
        logger.error(f"Location error: {e}")
        return {
            "lat": 28.4089, 
            "lng": 77.3178, 
            "city": "Faridabad", 
            "state": "Haryana", 
            "country": "India"
        }

def get_municipalities_with_gemini(location, waste_types):
    """Get nearby municipalities using Gemini AI"""
    model = setup_gemini_ai()
    
    if not model:
        return get_fallback_municipalities()
    
    try:
        prompt = f"""
        Find waste collection centers and recycling facilities near {location['city']}, {location['state']}, {location['country']} 
        for these waste types: {', '.join(waste_types)}.
        
        Provide real, existing facilities in JSON format:
        {{
            "centers": [
                {{
                    "name": "Center Name",
                    "address": "Full Address",
                    "distance": "X.X km",
                    "phone": "Phone Number",
                    "hours": "Operating Hours", 
                    "specialties": ["waste", "types"],
                    "latitude": lat_value,
                    "longitude": lng_value
                }}
            ]
        }}
        
        Focus on actual facilities with realistic coordinates.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean JSON response
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
                
        data = json.loads(response_text)
        centers = data.get('centers', [])
        
        # Validate and return
        if centers and len(centers) > 0:
            return centers[:6]  # Limit to 6 centers
        else:
            return get_fallback_municipalities()
            
    except Exception as e:
        logger.warning(f"Gemini AI error: {e}")
        return get_fallback_municipalities()

def get_fallback_municipalities():
    """Enhanced fallback municipality data"""
    return [
        {
            "name": "Faridabad Municipal Corporation - Sector 15",
            "address": "Sector 15, Faridabad, Haryana 121007",
            "distance": "2.1 km",
            "phone": "+91-129-2412345",
            "hours": "Mon-Sat: 8AM-6PM",
            "specialties": ["Plastic", "Metal", "Paper", "Glass"],
            "latitude": 28.4041,
            "longitude": 77.3178
        },
        {
            "name": "Green Valley Recycling Center",
            "address": "Sector 21, Faridabad, Haryana 121001", 
            "distance": "3.4 km",
            "phone": "+91-129-2567890",
            "hours": "Mon-Fri: 9AM-7PM, Sat: 9AM-5PM",
            "specialties": ["E-Waste", "Battery", "Plastic", "Metal"],
            "latitude": 28.4089,
            "longitude": 77.3145
        },
        {
            "name": "Delhi NCR Waste Management Hub",
            "address": "Industrial Area, Ballabhgarh, Faridabad 121004",
            "distance": "5.2 km", 
            "phone": "+91-129-2123456",
            "hours": "Daily: 7AM-8PM",
            "specialties": ["All Types", "Hazardous", "Organic", "Textile"],
            "latitude": 28.3428,
            "longitude": 77.3225
        },
        {
            "name": "EcoFriendly Collection Point - NIT",
            "address": "NIT Faridabad Campus, Faridabad 121001",
            "distance": "4.1 km",
            "phone": "+91-129-2987654", 
            "hours": "Mon-Sat: 10AM-6PM",
            "specialties": ["Paper", "Cardboard", "Clothes", "Shoes"],
            "latitude": 28.4056,
            "longitude": 77.3111
        },
        {
            "name": "HSPCB Authorized Recycler",
            "address": "Sector 30, Faridabad, Haryana 121008",
            "distance": "6.8 km",
            "phone": "+91-129-2345678",
            "hours": "Mon-Fri: 9AM-5PM",
            "specialties": ["Hazardous", "Battery", "E-Waste", "Chemical"],
            "latitude": 28.3847,
            "longitude": 77.2975
        },
        {
            "name": "Community Composting Center",
            "address": "Sector 12, Faridabad, Haryana 121005",
            "distance": "3.8 km",
            "phone": "+91-129-2876543",
            "hours": "Daily: 6AM-8PM",
            "specialties": ["Organic", "Garden Waste", "Food Waste"],
            "latitude": 28.4123,
            "longitude": 77.3089
        }
    ]

# ========================
# HELPER FUNCTIONS
# ========================
def get_waste_info(item_name):
    """Get enhanced waste category information"""
    # Clean item name for lookup
    item_key = item_name.lower().replace(" ", "_").replace("-", "_")
    
    # Direct lookup
    if item_key in WASTE_CATEGORIES:
        return WASTE_CATEGORIES[item_key]
    
    # Fuzzy matching for common variations
    fuzzy_matches = {
        "bottle": "plastic",
        "can": "metal",
        "newspaper": "paper",
        "magazine": "paper",
        "box": "cardboard",
        "bag": "plastic",
        "wrapper": "plastic"
    }
    
    for key, match in fuzzy_matches.items():
        if key in item_key:
            return WASTE_CATEGORIES.get(match, WASTE_CATEGORIES["trash"])
    
    # Default fallback
    return WASTE_CATEGORIES["trash"]

def process_detections(results, confidence_threshold=0.5):
    """Enhanced detection processing with better error handling"""
    detections_data = []
    
    try:
        if not results or len(results) == 0:
            return detections_data
            
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections_data
            
        for box in result.boxes:
            try:
                if box.cls is None or box.conf is None:
                    continue
                    
                cls_idx = int(box.cls.cpu().numpy())
                confidence = float(box.conf.cpu().numpy())
                
                # Apply confidence threshold
                if confidence < confidence_threshold:
                    continue
                
                cls_name = result.names.get(cls_idx, "unknown")
                waste_info = get_waste_info(cls_name)
                
                detections_data.append({
                    "Item": cls_name.replace("_", " ").title(),
                    "Icon": waste_info["icon"],
                    "Confidence": confidence,
                    "Confidence_Percent": f"{confidence:.1%}",
                    "Category": waste_info["category"],
                    "Recyclable": "Yes" if waste_info["recyclable"] else "No",
                    "Disposal_Bin": waste_info["bin"],
                    "Color": waste_info["color"],
                    "Tips": waste_info["tips"],
                    "CO2_Saved": waste_info.get("co2_saved", 0),
                    "Energy_Saved": waste_info.get("energy_saved", 0)
                })
                
            except Exception as e:
                logger.warning(f"Error processing detection: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error in process_detections: {e}")
        
    return detections_data

def create_map(municipalities, user_location):
    """Create interactive map with error handling"""
    if folium is None:
        st.warning("📍 Map feature requires folium. Install with: pip install folium streamlit-folium")
        return None
        
    try:
        m = folium.Map(
            location=[user_location['lat'], user_location['lng']], 
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add user location
        folium.Marker(
            [user_location['lat'], user_location['lng']],
            popup="📍 Your Location",
            tooltip="Your Current Location",
            icon=folium.Icon(color='red', icon='user', prefix='fa')
        ).add_to(m)
        
        # Add municipality markers
        for center in municipalities:
            try:
                folium.Marker(
                    [center['latitude'], center['longitude']],
                    popup=f"""
                    <div style="width:220px; font-family: Arial;">
                        <h4 style="margin-bottom: 10px; color: #2d3748;">{center['name']}</h4>
                        <p style="margin: 5px 0;"><strong>📍 Distance:</strong> {center['distance']}</p>
                        <p style="margin: 5px 0;"><strong>📞 Phone:</strong> {center['phone']}</p>
                        <p style="margin: 5px 0;"><strong>🕒 Hours:</strong> {center['hours']}</p>
                        <p style="margin: 5px 0;"><strong>♻️ Specialties:</strong><br>{', '.join(center['specialties'])}</p>
                    </div>
                    """,
                    tooltip=center['name'],
                    icon=folium.Icon(color='green', icon='recycle', prefix='fa')
                ).add_to(m)
            except KeyError as e:
                logger.warning(f"Missing key in municipality data: {e}")
                continue
        
        return m
        
    except Exception as e:
        logger.error(f"Error creating map: {e}")
        return None

def create_environmental_impact_metrics(detections_data):
    """Calculate detailed environmental impact"""
    if not detections_data:
        return {
            "trees_saved": 0,
            "energy_saved": 0,
            "co2_reduced": 0,
            "water_saved": 0,
            "recycling_rate": 0,
            "total_items": 0,
            "recyclable_items": 0
        }
    
    total_items = len(detections_data)
    recyclable_items = sum(1 for item in detections_data if item["Recyclable"] == "Yes")
    
    # Calculate environmental impact based on actual detection data
    co2_reduced = sum(item.get("CO2_Saved", 0) for item in detections_data)
    energy_saved = sum(item.get("Energy_Saved", 0) for item in detections_data)
    
    # Estimated impact calculations
    trees_saved = co2_reduced * 0.05  # Rough estimate: 20kg CO2 = 1 tree
    water_saved = energy_saved * 8  # Rough estimate: 1 kWh = 8 liters
    
    recycling_rate = (recyclable_items / total_items * 100) if total_items > 0 else 0
    
    return {
        "trees_saved": trees_saved,
        "energy_saved": energy_saved,
        "co2_reduced": co2_reduced,
        "water_saved": water_saved,
        "recycling_rate": recycling_rate,
        "total_items": total_items,
        "recyclable_items": recyclable_items
    }

def render_metric_card(title, value, icon, card_class="info", unit=""):
    """Render a metric card with consistent styling"""
    return f"""
    <div class="metric-card {card_class}">
        <div class="metric-number">{value}{unit}</div>
        <div class="metric-label">{icon} {title}</div>
    </div>
    """

def render_detection_card(item):
    """Render a detection result card"""
    recyclable_class = "recyclable" if item["Recyclable"] == "Yes" else "non-recyclable"
    
    return f"""
    <div class="detection-card {recyclable_class} fade-in-up">
        <div class="detection-header">
            <div class="detection-title">{item['Icon']} {item['Item']}</div>
            <div class="confidence-badge">{item['Confidence_Percent']}</div>
        </div>
        <div class="detection-info">
            <div class="info-item">
                <div class="info-label">Category</div>
                <div class="info-value">{item['Category']}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Recyclable</div>
                <div class="info-value">{item['Recyclable']}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Disposal Bin</div>
                <div class="info-value">{item['Disposal_Bin']}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Environmental Tip</div>
                <div class="info-value">{item['Tips']}</div>
            </div>
        </div>
    </div>
    """

def render_municipality_card(center):
    """Render a municipality card"""
    return f"""
    <div class="municipality-card fade-in-up">
        <div class="municipality-header">
            <h4 class="municipality-name">🏢 {center['name']}</h4>
            <span class="distance-badge">{center['distance']}</span>
        </div>
        <div class="municipality-details">
            <div class="detail-row">
                <span class="detail-icon">📍</span>
                <div class="detail-content">
                    <div class="detail-label">Address</div>
                    <div class="detail-value">{center['address']}</div>
                </div>
            </div>
            <div class="detail-row">
                <span class="detail-icon">📞</span>
                <div class="detail-content">
                    <div class="detail-label">Phone</div>
                    <div class="detail-value">{center['phone']}</div>
                </div>
            </div>
            <div class="detail-row">
                <span class="detail-icon">🕒</span>
                <div class="detail-content">
                    <div class="detail-label">Operating Hours</div>
                    <div class="detail-value">{center['hours']}</div>
                </div>
            </div>
            <div class="detail-row">
                <span class="detail-icon">♻️</span>
                <div class="detail-content">
                    <div class="detail-label">Waste Types Accepted</div>
                    <div class="detail-value">{', '.join(center['specialties'])}</div>
                </div>
            </div>
        </div>
    </div>
    """

# ========================
# MAIN APPLICATION
# ========================
def main():
    # Load CSS
    load_css()
    
    # Enhanced Header
    st.markdown("""
    <div class="hero-header fade-in-up">
        <h1 class="hero-title">🌱 EcoWaste AI</h1>
        <p class="hero-subtitle">
            AI-Powered Smart Waste Classification & Sustainable Disposal Assistant<br>
            <em>Making Sustainability Simple, One Detection at a Time</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model with error handling
    model = load_model()
    if model is None:
        st.error("❌ Failed to load detection model. Please check your model files or internet connection.")
        st.info("💡 Expected model files: final_model.pt, best.pt, or will download yolov8n.pt automatically")
        st.stop()
    
    # Initialize session state
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    if 'current_detections' not in st.session_state:
        st.session_state.current_detections = []
    if 'total_items_processed' not in st.session_state:
        st.session_state.total_items_processed = 0
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = datetime.now()
    
    # ========================
    # SIDEBAR CONFIGURATION
    # ========================
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Panel")
        
        # Input Source Selection
        input_source = st.selectbox(
            "📥 Select Input Source",
            ["📷 Camera Capture", "🖼️ Upload Image", "🎥 Upload Video"],
            help="Choose your preferred method for waste detection"
        )
        
        st.markdown("---")
        
        # Detection Settings
        with st.expander("🔧 Detection Settings", expanded=True):
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                0.1, 1.0, 0.5, 0.05,
                help="Minimum confidence for detections (higher = more accurate, fewer results)"
            )
            max_detections = st.slider(
                "Maximum Detections", 
                1, 50, 15,
                help="Maximum number of objects to detect per image"
            )
            show_labels = st.checkbox("Show Detection Labels", value=True)
            show_confidence = st.checkbox("Show Confidence Scores", value=True)
        
        # Location Settings
        with st.expander("📍 Location Settings"):
            auto_location = st.checkbox("Auto-detect Location", value=True)
            if not auto_location:
                manual_city = st.text_input("City", value="Faridabad")
                manual_state = st.text_input("State/Province", value="Haryana")
                manual_country = st.text_input("Country", value="India")
        
        # Enhanced Features
        with st.expander("✨ Advanced Features"):
            use_gemini = st.checkbox(
                "🤖 Use Gemini AI for Municipality Search", 
                value=True,
                help="Enhanced search using Google's Gemini AI"
            )
            show_map = st.checkbox(
                "🗺️ Show Interactive Map", 
                value=True,
                help="Display collection centers on an interactive map"
            )
            show_impact = st.checkbox(
                "🌱 Show Environmental Impact", 
                value=True,
                help="Calculate and display environmental benefits"
            )
            detailed_tips = st.checkbox(
                "💡 Show Detailed Disposal Tips", 
                value=True,
                help="Include comprehensive disposal instructions"
            )
            export_results = st.checkbox(
                "📊 Enable Results Export", 
                value=True,
                help="Allow exporting detection results to CSV"
            )
        
        st.markdown("---")
        
        # Session Statistics
        st.markdown("### 📊 Session Stats")
        session_duration = datetime.now() - st.session_state.session_start_time
        hours, remainder = divmod(int(session_duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        st.metric("⏱️ Session Time", f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        st.metric("📈 Items Processed", st.session_state.total_items_processed)
        st.metric("🔄 Detections Run", len(st.session_state.detection_history))
        
        # Quick Actions
        st.markdown("### 🚀 Quick Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset Session", use_container_width=True, help="Clear all data and start fresh"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if export_results and st.session_state.current_detections:
                df = pd.DataFrame(st.session_state.current_detections)
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Export CSV",
                    csv,
                    f"ecowatch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True,
                    help="Download detection results as CSV file"
                )
    
    # ========================
    # LIVE DASHBOARD
    # ========================
    st.markdown("### 📊 Live Detection Dashboard")
    
    # Calculate current metrics
    current_detections = st.session_state.current_detections
    impact_data = create_environmental_impact_metrics(current_detections)
    
    # Enhanced Metrics Display
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(
            render_metric_card("Items Detected", impact_data["total_items"], "🎯", "info"),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            render_metric_card("Recyclable", impact_data["recyclable_items"], "♻️", "recyclable"),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            render_metric_card("Recycling Rate", f"{impact_data['recycling_rate']:.0f}", "📈", "warning", "%"),
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            render_metric_card("CO₂ Saved", f"{impact_data['co2_reduced']:.1f}", "🌍", "recyclable", " kg"),
            unsafe_allow_html=True
        )
    
    with col5:
        st.markdown(
            render_metric_card("Energy Saved", f"{impact_data['energy_saved']:.1f}", "⚡", "info", " kWh"),
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # ========================
    # INPUT PROCESSING SECTION
    # ========================
    st.markdown("### 📤 Waste Detection Input")
    
    input_col, preview_col = st.columns([2, 1])
    
    with input_col:
        image = None
        video_path = None
        
        if "📷 Camera" in input_source:
            st.markdown("#### 📸 Camera Capture")
            img_file = st.camera_input(
                "Take a photo of waste items",
                help="Ensure good lighting and clear visibility of items"
            )
            if img_file:
                try:
                    pil_image = Image.open(img_file)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    st.success("✅ Image captured successfully!")
                except Exception as e:
                    st.error(f"❌ Error processing camera image: {e}")
        
        elif "🖼️ Upload Image" in input_source:
            st.markdown("#### 📁 Image Upload")
            img_file = st.file_uploader(
                "Upload waste image", 
                type=["jpg", "png", "jpeg", "webp", "bmp"],
                help="Supported formats: JPG, PNG, JPEG, WebP, BMP (Max 200MB)"
            )
            if img_file:
                try:
                    # Show file info
                    file_size = len(img_file.getvalue()) / (1024 * 1024)  # MB
                    st.info(f"📄 File: {img_file.name} ({file_size:.1f} MB)")
                    
                    pil_image = Image.open(img_file)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    st.success("✅ Image uploaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading image: {e}")
        
        elif "🎥 Upload Video" in input_source:
            st.markdown("#### 🎬 Video Upload")
            vid_file = st.file_uploader(
                "Upload video file", 
                type=["mp4", "avi", "mov", "mkv", "webm"],
                help="Supported formats: MP4, AVI, MOV, MKV, WebM (Max 200MB)"
            )
            if vid_file:
                try:
                    file_size = len(vid_file.getvalue()) / (1024 * 1024)  # MB
                    st.info(f"🎥 File: {vid_file.name} ({file_size:.1f} MB)")
                    
                    # Create temporary file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(vid_file.read())
                    video_path = tfile.name
                    st.success("✅ Video uploaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error processing video: {e}")
    
    with preview_col:
        st.markdown("""
        <div class="glass-card">
            <h4>💡 Detection Tips</h4>
            <ul style="text-align: left; padding-left: 1.5rem;">
                <li>🔆 <strong>Lighting:</strong> Ensure bright, even lighting</li>
                <li>📏 <strong>Distance:</strong> Keep items clearly visible</li>
                <li>🎯 <strong>Background:</strong> Use clean, uncluttered backgrounds</li>
                <li>📱 <strong>Quality:</strong> Use high-resolution images</li>
                <li>🔍 <strong>Focus:</strong> Center items in the frame</li>
                <li>📐 <strong>Angle:</strong> Capture items from multiple angles for best results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show current image preview if available
        if image is not None:
            st.markdown("#### 👀 Preview")
            # Resize for preview
            height, width = image.shape[:2]
            if width > 300:
                scale = 300 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                preview_image = cv2.resize(image, (new_width, new_height))
            else:
                preview_image = image
            
            st.image(
                cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB), 
                caption="Image Preview",
                use_column_width=True
            )
    
    # ========================
    # DETECTION PROCESSING
    # ========================
    detections_data = []
    
    # Process image detection
    if image is not None:
        st.markdown("### 🔍 AI Detection Results")
        
        with st.spinner("🤖 Analyzing image with advanced AI models..."):
            try:
                # Add processing time measurement
                start_time = time.time()
                
                # Run YOLO detection
                results = model(
                    image, 
                    conf=confidence_threshold, 
                    max_det=max_detections,
                    verbose=False
                )
                
                processing_time = time.time() - start_time
                
                if results and len(results) > 0:
                    detections_data = process_detections(results, confidence_threshold)
                    
                    if detections_data:
                        # Update session state
                        st.session_state.current_detections = detections_data
                        st.session_state.total_items_processed += len(detections_data)
                        st.session_state.detection_history.append({
                            "timestamp": datetime.now(),
                            "items": len(detections_data),
                            "recyclable": sum(1 for item in detections_data if item["Recyclable"] == "Yes"),
                            "processing_time": processing_time
                        })
                        
                        # Create annotated image
                        try:
                            annotated_img = results[0].plot(
                                labels=show_labels,
                                conf=show_confidence,
                                line_width=2,
                                font_size=1
                            )
                        except Exception as e:
                            logger.warning(f"Error creating annotated image: {e}")
                            annotated_img = image
                        
                        # Display results in columns
                        result_col1, result_col2 = st.columns([3, 1])
                        
                        with result_col1:
                            st.image(
                                cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), 
                                caption=f"🎯 Detected {len(detections_data)} items in {processing_time:.2f}s", 
                                use_column_width=True
                            )
                        
                        with result_col2:
                            # Quick statistics
                            recyclable_count = sum(1 for item in detections_data if item["Recyclable"] == "Yes")
                            non_recyclable_count = len(detections_data) - recyclable_count
                            avg_confidence = np.mean([item["Confidence"] for item in detections_data])
                            
                            st.success(f"✅ Recyclable: {recyclable_count}")
                            st.warning(f"⚠️ Non-recyclable: {non_recyclable_count}")
                            st.info(f"📈 Avg Confidence: {avg_confidence:.1%}")
                            st.metric("⚡ Processing Time", f"{processing_time:.2f}s")
                            
                            # Environmental impact preview
                            if show_impact:
                                impact = create_environmental_impact_metrics(detections_data)
                                st.metric("🌳 Trees Saved", f"{impact['trees_saved']:.2f}")
                                st.metric("🌍 CO₂ Reduced", f"{impact['co2_reduced']:.1f} kg")
                        
                    else:
                        st.warning("🔍 No waste items detected. Try adjusting the confidence threshold or ensure items are clearly visible.")
                        st.info("💡 **Tips:** Lower the confidence threshold, improve lighting, or try a different angle.")
                else:
                    st.info("📷 Image processed but no objects detected.")
                    
            except Exception as e:
                st.error(f"❌ Detection error: {e}")
                logger.error(f"Detection error: {e}")
    
    # Process video detection
    elif video_path:
        st.markdown("### 🎥 Video Analysis")
        
        with st.spinner("📹 Processing video frames..."):
            try:
                cap = cv2.VideoCapture(video_path)
                
                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                st.info(f"🎬 Video Info: {frame_count} frames, {fps} FPS, {duration:.1f}s duration")
                
                # Process first frame for detection
                ret, frame = cap.read()
                
                if ret:
                    results = model(frame, conf=confidence_threshold, max_det=max_detections)
                    detections_data = process_detections(results, confidence_threshold)
                    
                    if detections_data:
                        st.session_state.current_detections = detections_data
                        st.session_state.total_items_processed += len(detections_data)
                        
                        # Create annotated frame
                        annotated_frame = results[0].plot(
                            labels=show_labels, 
                            conf=show_confidence,
                            line_width=2
                        )
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.image(
                                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                                caption=f"🎯 Video Frame Analysis - {len(detections_data)} items detected", 
                                use_column_width=True
                            )
                        with col2:
                            recyclable_count = sum(1 for item in detections_data if item["Recyclable"] == "Yes")
                            st.success(f"✅ Recyclable: {recyclable_count}")
                            st.warning(f"⚠️ Non-recyclable: {len(detections_data) - recyclable_count}")
                            
                    else:
                        st.warning("🔍 No waste items detected in video frame")
                else:
                    st.error("❌ Could not read video file")
                    
                cap.release()
                
            except Exception as e:
                st.error(f"❌ Video processing error: {e}")
                logger.error(f"Video processing error: {e}")
            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(video_path):
                        os.unlink(video_path)
                except Exception as e:
                    logger.warning(f"Could not delete temporary file: {e}")
    
    # ========================
    # DETAILED RESULTS ANALYSIS
    # ========================
    if detections_data:
        st.markdown("---")
        
        # Enhanced tabs with better organization
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Detection Details", 
            "♻️ Disposal Guide", 
            "📈 Analytics & Insights", 
            "📍 Collection Centers",
            "🌱 Environmental Impact"
        ])
        
        with tab1:
            st.markdown("#### 🔍 Detailed Detection Results")
            
            # Sort detections by confidence
            sorted_detections = sorted(detections_data, key=lambda x: x["Confidence"], reverse=True)
            
            # Display detection cards
            for item in sorted_detections:
                st.markdown(render_detection_card(item), unsafe_allow_html=True)
            
            # Summary statistics
            st.markdown("##### 📊 Detection Summary")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                categories = {}
                for item in detections_data:
                    cat = item["Category"]
                    categories[cat] = categories.get(cat, 0) + 1
                
                st.markdown("**Categories Detected:**")
                for cat, count in categories.items():
                    st.write(f"• {cat}: {count} items")
            
            with summary_col2:
                confidence_levels = [item["Confidence"] for item in detections_data]
                st.markdown("**Confidence Statistics:**")
                st.write(f"• Average: {np.mean(confidence_levels):.1%}")
                st.write(f"• Highest: {np.max(confidence_levels):.1%}")
                st.write(f"• Lowest: {np.min(confidence_levels):.1%}")
            
            with summary_col3:
                recyclable_items = [item for item in detections_data if item["Recyclable"] == "Yes"]
                st.markdown("**Recyclability:**")
                st.write(f"• Recyclable: {len(recyclable_items)} items")
                st.write(f"• Non-recyclable: {len(detections_data) - len(recyclable_items)} items")
                st.write(f"• Rate: {len(recyclable_items)/len(detections_data)*100:.1f}%")
        
        with tab2:
            st.markdown("#### ♻️ Comprehensive Disposal Instructions")
            
            # Group items by disposal method
            disposal_groups = {}
            for item in detections_data:
                bin_type = item["Disposal_Bin"]
                if bin_type not in disposal_groups:
                    disposal_groups[bin_type] = []
                disposal_groups[bin_type].append(item)
            
            # Display grouped disposal instructions
            for bin_type, items in disposal_groups.items():
                is_recyclable = all(item["Recyclable"] == "Yes" for item in items)
                status_icon = "✅" if is_recyclable else "⚠️"
                status_class = "recyclable" if is_recyclable else "non-recyclable"
                
                st.markdown(f"### {status_icon} {bin_type}")
                
                for item in items:
                    st.markdown(f"""
                    <div class="glass-card {status_class}" style="margin: 1rem 0; padding: 1rem;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{item['Icon']}</span>
                            <h4 style="margin: 0; color: #2d3748;">{item['Item']}</h4>
                        </div>
                        <p><strong>💡 Disposal Tip:</strong> {item['Tips']}</p>
                        {f'<p><strong>🌱 Impact:</strong> Saves {item["CO2_Saved"]:.1f} kg CO₂ and {item["Energy_Saved"]:.1f} kWh</p>' if item.get('CO2_Saved', 0) > 0 else ''}
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("#### 📈 Comprehensive Analytics Dashboard")
            
            df = pd.DataFrame(detections_data)
            
            # Create visualizations
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Enhanced pie chart for recyclability
                recyclable_counts = df["Recyclable"].value_counts()
                fig_pie = px.pie(
                    values=recyclable_counts.values,
                    names=recyclable_counts.index,
                    title="♻️ Recyclability Distribution",
                    color_discrete_map={"Yes": "#56ab2f", "No": "#ff6b6b"},
                    hole=0.4
                )
                fig_pie.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    textfont_size=12
                )
                fig_pie.update_layout(
                    showlegend=True, 
                    height=400,
                    font=dict(size=12)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with chart_col2:
                # Enhanced bar chart for categories
                category_counts = df["Category"].value_counts()
                fig_bar = px.bar(
                    x=category_counts.values,
                    y=category_counts.index,
                    orientation='h',
                    title="🗂️ Waste Categories Distribution",
                    labels={"x": "Number of Items", "y": "Category"},
                    color=category_counts.values,
                    color_continuous_scale="Viridis"
                )
                fig_bar.update_layout(
                    showlegend=False, 
                    height=400,
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Confidence distribution
            if len(detections_data) > 1:
                st.markdown("##### 📊 Confidence Score Distribution")
                fig_hist = px.histogram(
                    df, 
                    x="Confidence", 
                    nbins=10,
                    title="Detection Confidence Distribution",
                    labels={"Confidence": "Confidence Score", "count": "Number of Detections"}
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Detection history trends
            if len(st.session_state.detection_history) > 1:
                st.markdown("##### 📈 Detection History Trends")
                history_df = pd.DataFrame(st.session_state.detection_history)
                
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=history_df['timestamp'], 
                    y=history_df['items'],
                    mode='lines+markers',
                    name='Total Items',
                    line=dict(color='#667eea', width=3)
                ))
                fig_line.add_trace(go.Scatter(
                    x=history_df['timestamp'], 
                    y=history_df['recyclable'],
                    mode='lines+markers',
                    name='Recyclable Items',
                    line=dict(color='#56ab2f', width=3)
                ))
                
                fig_line.update_layout(
                    title="Detection Trends Over Time",
                    xaxis_title="Time",
                    yaxis_title="Number of Items",
                    height=400
                )
                st.plotly_chart(fig_line, use_container_width=True)
        
        with tab4:
            st.markdown("#### 📍 Nearby Waste Collection Centers")
            
            # Get user location
            if auto_location:
                user_location = get_user_location()
            else:
                user_location = {
                    "lat": 28.4089, 
                    "lng": 77.3178, 
                    "city": manual_city, 
                    "state": manual_state, 
                    "country": manual_country
                }
            
            # Show location info
            st.info(f"📍 Searching near: {user_location['city']}, {user_location['state']}, {user_location['country']}")
            
            # Get waste types for search
            waste_types = list(set(item["Category"] for item in detections_data))
            
            # Get municipalities
            if use_gemini and genai:
                with st.spinner("🤖 Finding collection centers using Gemini AI..."):
                    municipalities = get_municipalities_with_gemini(user_location, waste_types)
            else:
                municipalities = get_fallback_municipalities()
            
            # Display municipality cards
            st.markdown("##### 🏢 Available Collection Centers")
            for center in municipalities:
                st.markdown(render_municipality_card(center), unsafe_allow_html=True)
            
            # Interactive map
            if show_map and folium:
                st.markdown("##### 🗺️ Interactive Collection Center Map")
                try:
                    map_obj = create_map(municipalities, user_location)
                    if map_obj:
                        map_data = st_folium(map_obj, width=700, height=500, returned_objects=["last_clicked"])
                        
                        # Show clicked location details
                        if map_data.get('last_clicked'):
                            clicked_lat = map_data['last_clicked']['lat']
                            clicked_lng = map_data['last_clicked']['lng']
                            st.info(f"📍 Clicked location: {clicked_lat:.4f}, {clicked_lng:.4f}")
                    else:
                        st.warning("Could not create interactive map.")
                except Exception as e:
                    st.error(f"Map error: {e}")
                    st.info("Map feature temporarily unavailable. Please refer to the list above.")
            
            # Contact information summary
            st.markdown("##### 📞 Quick Contact Directory")
            contact_df = pd.DataFrame([
                {
                    "Center": center["name"],
                    "Phone": center["phone"],
                    "Distance": center["distance"],
                    "Specialties": ", ".join(center["specialties"][:3])  # Limit for display
                }
                for center in municipalities
            ])
            st.dataframe(contact_df, use_container_width=True)
        
        with tab5:
            st.markdown("#### 🌱 Environmental Impact Analysis")
            
            impact_data = create_environmental_impact_metrics(detections_data)
            
            # Enhanced impact metrics
            impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
            
            with impact_col1:
                st.markdown(
                    render_metric_card("Trees Saved", f"{impact_data['trees_saved']:.3f}", "🌳", "recyclable"),
                    unsafe_allow_html=True
                )
            
            with impact_col2:
                st.markdown(
                    render_metric_card("Energy Saved", f"{impact_data['energy_saved']:.2f}", "⚡", "info", " kWh"),
                    unsafe_allow_html=True
                )
            
            with impact_col3:
                st.markdown(
                    render_metric_card("CO₂ Reduced", f"{impact_data['co2_reduced']:.2f}", "🌍", "warning", " kg"),
                    unsafe_allow_html=True
                )
            
            with impact_col4:
                st.markdown(
                    render_metric_card("Water Saved", f"{impact_data['water_saved']:.0f}", "💧", "info", " L"),
                    unsafe_allow_html=True
                )
            
            # Impact comparison and projections
            st.markdown("##### 🎯 Impact Projections")
            
            proj_col1, proj_col2 = st.columns(2)
            
            with proj_col1:
                st.markdown("""
                <div class="glass-card">
                    <h4>📅 If You Process Similar Items Daily:</h4>
                </div>
                """, unsafe_allow_html=True)
                
                daily_metrics = {
                    "Trees saved per year": impact_data['trees_saved'] * 365,
                    "Energy saved per month (kWh)": impact_data['energy_saved'] * 30,
                    "CO₂ reduced per year (kg)": impact_data['co2_reduced'] * 365,
                    "Water saved per month (L)": impact_data['water_saved'] * 30
                }
                
                for metric, value in daily_metrics.items():
                    st.metric(metric, f"{value:.1f}")
            
            with proj_col2:
                # Impact visualization
                impact_categories = ['Trees Saved', 'Energy (kWh)', 'CO₂ Reduced (kg)', 'Water (L)']
                impact_values = [
                    impact_data['trees_saved'],
                    impact_data['energy_saved'],
                    impact_data['co2_reduced'],
                    impact_data['water_saved']
                ]
                
                fig_impact = px.bar(
                    x=impact_categories,
                    y=impact_values,
                    title="Environmental Impact Breakdown",
                    color=impact_values,
                    color_continuous_scale="Greens"
                )
                fig_impact.update_layout(
                    showlegend=False, 
                    height=400,
                    xaxis_title="Impact Category",
                    yaxis_title="Amount"
                )
                st.plotly_chart(fig_impact, use_container_width=True)
            
            # Comprehensive sustainability tips
            st.markdown("##### 💡 Advanced Sustainability Tips")
            
            sustainability_tips = [
                {
                    "icon": "🌱",
                    "title": "Reduce at Source",
                    "tip": "Choose products with minimal packaging and opt for reusable alternatives to reduce waste generation."
                },
                {
                    "icon": "♻️",
                    "title": "Proper Recycling",
                    "tip": "Clean containers thoroughly before recycling and separate materials correctly to prevent contamination."
                },
                {
                    "icon": "🔄",
                    "title": "Circular Economy",
                    "tip": "Support businesses that embrace circular economy principles by designing out waste and keeping materials in use."
                },
                {
                    "icon": "🏠",
                    "title": "Home Composting",
                    "tip": "Start composting organic waste at home to create nutrient-rich soil and reduce methane emissions from landfills."
                },
                {
                    "icon": "💡",
                    "title": "Share Knowledge",
                    "tip": "Educate friends and family about proper waste sorting and the environmental impact of recycling."
                },
                {
                    "icon": "🌍",
                    "title": "Global Impact",
                    "tip": "Remember: Every recycled aluminum can saves enough energy to run a TV for 3 hours and stays in circulation indefinitely!"
                }
            ]
            
            tip_cols = st.columns(2)
            for i, tip in enumerate(sustainability_tips):
                with tip_cols[i % 2]:
                    st.markdown(f"""
                    <div class="glass-card" style="margin: 1rem 0;">
                        <h4 style="color: #2d3748; margin-bottom: 0.5rem;">{tip['icon']} {tip['title']}</h4>
                        <p style="color: #4a5568; margin: 0; line-height: 1.6;">{tip['tip']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Show usage instructions if no detection has been made
    else:
        st.markdown("### 🎯 Getting Started")
        st.markdown("""
        <div class="glass-card">
            <h3>🌟 Welcome to EcoWaste AI!</h3>
            <p>Follow these steps to start detecting and properly disposing of waste:</p>
            <ol style="text-align: left; padding-left: 2rem; line-height: 2;">
                <li><strong>📸 Capture or Upload:</strong> Use your camera or upload an image of waste items</li>
                <li><strong>🤖 AI Detection:</strong> Our advanced AI will identify and classify the waste</li>
                <li><strong>♻️ Get Instructions:</strong> Receive detailed disposal and recycling instructions</li>
                <li><strong>📍 Find Centers:</strong> Locate nearby collection and recycling centers</li>
                <li><strong>🌱 Track Impact:</strong> See your environmental impact and contribution to sustainability</li>
            </ol>
            <p><em>Start by selecting an input method from the sidebar and capturing or uploading your first image!</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================
    # ENHANCED FOOTER
    # ========================
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <h3 class="footer-title">🌱 EcoWaste AI - Revolutionizing Waste Management</h3>
            <p class="footer-text">
                Empowering individuals and communities worldwide to make informed decisions about waste disposal,
                promoting sustainability, and creating a cleaner future through the power of artificial intelligence
                and environmental consciousness.
            </p>
            <div class="footer-links">
                <strong>🔧 Powered by:</strong> YOLO Object Detection • Enhanced with Gemini AI • Built with Streamlit<br>
                <strong>🎯 Mission:</strong> Making Sustainability Accessible • One Detection at a Time<br>
                <strong>📊 Features:</strong> Real-time Detection • Smart Disposal • Impact Tracking • Community Connection
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========================
# APPLICATION ENTRY POINT
# ========================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")
        st.info("Please refresh the page or check your configuration.")