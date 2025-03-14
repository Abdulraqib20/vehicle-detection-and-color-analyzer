import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import tempfile
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras import layers

DepthwiseConv2D = layers.DepthwiseConv2D

st.set_page_config(page_icon="🚗", page_title="Vehicle Type Detection", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    h1, h2, h3 {
        color: #0a58ca;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1 {
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 15px;
        border-bottom: 2px solid #0a58ca;
    }
    .sidebar .sidebar-content {
        background-color: #212529;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .vehicle-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #0d6efd;
    }
    .stats-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 15px 0;
    }
    .stat-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px 15px;
        min-width: 120px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stat-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #0d6efd;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #6c757d;
    }
    .color-swatch {
        display: inline-block;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        margin-right: 10px;
        vertical-align: middle;
        border: 2px solid white;
        box-shadow: 0 0 0 1px rgba(0,0,0,0.1);
    }
    .color-prediction {
        display: flex;
        align-items: center;
        margin: 5px 0;
        padding: 8px 12px;
        background-color: #f8f9fa;
        border-radius: 6px;
    }
    .color-name {
        font-weight: 500;
        margin-right: 10px;
    }
    .color-confidence {
        margin-left: auto;
        background-color: #0d6efd;
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d6efd !important;
        color: white !important;
    }
    .upload-section {
        background-color: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 30px;
        text-align: center;
        border: 2px dashed #dee2e6;
    }
    .metrics-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
    }
    .metrics-icon {
        font-size: 24px;
        margin-right: 12px;
        color: #0d6efd;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Add these styles to your existing CSS */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f8f9fa;
        padding: 10px 20px;
        text-align: center;
        font-size: 0.9rem;
        border-top: 1px solid #dee2e6;
        z-index: 1000;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }
    .footer a {
        color: #0d6efd;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .author-badge {
        display: inline-block;
        background-color: #e9ecef;
        border-radius: 20px;
        padding: 5px 15px;
        margin: 10px 0;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# Color class configuration
class_subset = ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 
                'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']

# Custom layer for color model
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)


# Load models with caching
@st.cache_resource
def load_models():
    yolo_model = YOLO('models/yolov8m-seg.pt')
    
    # Load color model with custom layer
    color_model = tf.keras.models.load_model(
        'models/EFN-model.best.h5',
        custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}
    )
    
    return yolo_model, color_model

def predict_color(cropped_image, color_model):
    """Predict color from cropped vehicle image"""
    try:
        # Preprocess image for color model
        img = cv2.resize(cropped_image, (224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        predictions = color_model.predict(img_array, verbose=0)[0]
        sorted_indices = np.argsort(predictions)[::-1]
        
        return [
            {
                'color': class_subset[idx],
                'confidence': float(predictions[idx])
            } for idx in sorted_indices if predictions[idx] > 0.1
        ]
    except Exception as e:
        st.error(f"Color prediction error: {str(e)}")
        return []

def process_image(image_np, yolo_model, color_model):
    """Process image and return predictions with colors"""
    # Important fix: Use copy of original image for detection to avoid analyzing annotated image
    original_image = image_np.copy()
    
    results = yolo_model(original_image)
    annotated_image = results[0].plot()  # Create annotated image for display
    
    detections = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = yolo_model.names[class_id]
            confidence = float(box.conf)
            
            if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                # Crop vehicle from ORIGINAL image
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped_vehicle = original_image[y1:y2, x1:x2]
                
                # Get color predictions
                color_predictions = predict_color(cropped_vehicle, color_model)
                
                detections.append({
                    'class': class_name.upper(),
                    'confidence': confidence,
                    'colors': color_predictions,
                    'box': [x1, y1, x2, y2]
                })
    
    return annotated_image, detections

def main():
    st.title("🚗 Vehicle Detection & Color Analyzer")
    
    # Load models
    with st.spinner("Loading AI models..."):
        yolo_model, color_model = load_models()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        
        st.subheader("Detection Parameters")
        confidence_threshold = st.slider(
            "Detection Confidence",
            min_value=0.0, max_value=1.0, value=0.5, 
            format="%.2f",
            help="Minimum confidence threshold for vehicle detection"
        )
        
        st.subheader("Display Options")
        show_bounding_boxes = st.checkbox("Show bounding boxes", value=True)
        show_color_swatches = st.checkbox("Show color predictions", value=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📋 About
        This application uses AI to:
        - Detect vehicles in images
        - Identify vehicle types
        - Predict vehicle colors
        
        Upload your images to get started!
        """)
    
    # File upload section
    with st.container():
        st.markdown("""
        <div class="upload-section">
            <h3>📤 Upload Vehicle Images</h3>
            <p>Upload images containing vehicles to analyze their type and color</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

    if uploaded_files:
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert('RGB'))
            
            # Process image
            start_time = time.time()
            annotated_image, vehicles = process_image(image_np, yolo_model, color_model)
            processing_time = time.time() - start_time
            
            # Display results in tabs
            st.markdown(f"""
            <h2 style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6;">
                Analysis Results: {uploaded_file.name}
            </h2>
            """, unsafe_allow_html=True)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"""
                    <div class="metrics-card">
                        <div class="metrics-icon">🔍</div>
                        <div>
                            <div style="font-size: 0.9rem; color: #6c757d;">Vehicles Detected</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">{len(vehicles)}</div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="metrics-card">
                        <div class="metrics-icon">⏱️</div>
                        <div>
                            <div style="font-size: 0.9rem; color: #6c757d;">Processing Time</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">{processing_time:.2f}s</div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with col3:
                most_common_type = max(vehicles, key=lambda x: x['confidence'])['class'] if vehicles else "N/A"
                st.markdown(
                    f"""
                    <div class="metrics-card">
                        <div class="metrics-icon">🚙</div>
                        <div>
                            <div style="font-size: 0.9rem; color: #6c757d;">Primary Vehicle Type</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">{most_common_type}</div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Image display tabs
            tab1, tab2 = st.tabs(["📸 Original Image", "🔍 Analyzed Image"])
            
            with tab1:
                st.image(image, use_column_width=True)
            
            with tab2:
                st.image(annotated_image, use_column_width=True)
            
            # Vehicle details
            if vehicles:
                st.markdown("### 🚗 Vehicle Analysis")
                
                for idx, vehicle in enumerate(vehicles, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="vehicle-card">
                            <h4>Vehicle {idx}: {vehicle['class']}</h4>
                        """, unsafe_allow_html=True)
                        
                        # Vehicle stats in columns
                        vcol1, vcol2 = st.columns(2)
                        
                        with vcol1:
                            st.markdown(f"""
                            <div class="stat-box">
                                <div class="stat-value">{vehicle['confidence']*100:.1f}%</div>
                                <div class="stat-label">Detection Confidence</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with vcol2:
                            # Get primary color (highest confidence)
                            primary_color = vehicle['colors'][0]['color'] if vehicle['colors'] else "Unknown"
                            st.markdown(f"""
                            <div class="stat-box">
                                <div class="stat-value" style="display: flex; align-items: center; justify-content: center;">
                                    <div class="color-swatch" style="background-color: {primary_color.lower()}"></div>
                                    {primary_color.upper()}
                                </div>
                                <div class="stat-label">Primary Color</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show detected colors
                        st.markdown("<h5>🎨 Color Analysis</h5>", unsafe_allow_html=True)
                        
                        # Extract vehicle image for display
                        x1, y1, x2, y2 = vehicle['box']
                        cropped_vehicle = image_np[y1:y2, x1:x2]
                        
                        # Display cropped vehicle and color predictions side by side
                        ccol1, ccol2 = st.columns([1, 2])
                        
                        with ccol1:
                            st.image(cropped_vehicle, caption=f"{vehicle['class']} Crop", use_column_width=True)
                        
                        with ccol2:
                            # Display color predictions with progress bars
                            for color in vehicle['colors'][:4]:  # Show top 4 colors
                                st.markdown(f"""
                                <div class="color-prediction">
                                    <div class="color-swatch" style="background-color: {color['color'].lower()}"></div>
                                    <div class="color-name">{color['color'].upper()}</div>
                                    <div class="color-confidence">{color['confidence']*100:.1f}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.progress(min(color['confidence'], 1.0))
                        
                        st.markdown("</div>", unsafe_allow_html=True)  # Close vehicle-card div
            else:
                st.warning("⚠️ No vehicles detected in this image")
            
            # Download options
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                Image.fromarray(annotated_image).save(tmp.name)
                tmp.close()
                
            try:
                with open(tmp.name, "rb") as file:
                    st.download_button(
                        label="💾 Download Analyzed Image",
                        data=file,
                        file_name=f"analyzed_{uploaded_file.name}",
                        mime="image/jpeg"
                    )
            finally:
                if os.path.exists(tmp.name):
                    try:
                        os.unlink(tmp.name)
                    except PermissionError:
                        time.sleep(0.5)
                        os.unlink(tmp.name)
            
            # Update progress
            progress_bar.progress((i+1)/len(uploaded_files))
            
            # Separator between multiple images
            if i < len(uploaded_files) - 1:
                st.markdown("<hr>", unsafe_allow_html=True)
        
        # Final success message
        st.success("✅ Analysis Complete!")
        st.balloons()
    else:
        # Show placeholder when no images are uploaded
        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 10px; padding: 30px; text-align: center; margin-top: 30px;">
            <img src="https://img.icons8.com/fluency/96/000000/car.png" style="width: 60px; height: 60px; margin-bottom: 15px;">
            <h3>No Images Uploaded Yet</h3>
            <p>Upload vehicle images above to begin analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="author-badge">
        <span>👨‍💻 Built with 💖 by <a href="https://github.com/Abdulraqib20" target="_blank">raqibcodes</a></span>
    </div>
    
    <div class="footer">
        <div>🚗 Vehicle Detection & Color Identification</div>
        <div>© 2025 <a href="https://github.com/Abdulraqib20" target="_blank">raqibcodes</a> | All Rights Reserved</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:100px'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()