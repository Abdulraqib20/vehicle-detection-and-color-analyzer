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

st.set_page_config(page_icon="🚗", page_title="Vehicle Type Detection")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: #2d2d2d;
    }
    .color-swatch {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 10px;
        vertical-align: middle;
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
    results = yolo_model(image_np)
    detections = []
    
    for result in results:
        annotated_image = result.plot()
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = yolo_model.names[class_id]
            confidence = float(box.conf)
            
            if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                # Crop vehicle from image
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped_vehicle = image_np[y1:y2, x1:x2]
                
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
    st.title("🚗🔍 Vehicle Detection & Color Analyzer")
    st.markdown("### AI-Powered Vehicle Recognition System")
    
    # Load models
    yolo_model, color_model = load_models()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.0, max_value=1.0, value=0.5
        )
        st.markdown("---")
        st.info("Upload images of vehicles to analyze their type and color characteristics.")
    
    # File upload
    uploaded_files = st.file_uploader(
        "📤 Upload Vehicle Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
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
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(annotated_image, caption="Analyzed Image", use_column_width=True)
            
            # Detailed analysis
            with st.expander(f"📊 Detailed Analysis for {uploaded_file.name}", expanded=True):
                if vehicles:
                    st.markdown(f"""
                        <div class='result-box'>
                            <h4>🚦 Detection Summary</h4>
                            <p>Total Vehicles Detected: {len(vehicles)}</p>
                            <p>Processing Time: {processing_time:.2f}s</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    for idx, vehicle in enumerate(vehicles, 1):
                        color_items = "".join(
                            f"""<div style="margin: 5px 0;">
                                <div class="color-swatch" style="background-color: {color['color'].lower()}"></div>
                                {color['color'].upper()}: {color['confidence']:.1%}
                            </div>"""
                            for color in vehicle['colors'][:3]  # Show top 3 colors
                        )
                        
                        st.markdown(f"""
                            <div class='result-box'>
                                <h4>🚙 Vehicle {idx}</h4>
                                <p>Type: {vehicle['class']}</p>
                                <p>Confidence: {vehicle['confidence']*100:.1f}%</p>
                                <h5 style="margin-top: 15px;">🎨 Predicted Colors:</h5>
                                {color_items}
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No vehicles detected in this image")
            
            # Download handling
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                Image.fromarray(annotated_image).save(tmp.name)
                tmp.close()
                
            try:
                with open(tmp.name, "rb") as file:
                    btn = st.download_button(
                        label=f"💾 Download Analyzed Image {i+1}",
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
            
            progress_bar.progress((i+1)/len(uploaded_files))
        
        st.success("✅ Analysis Complete!")
        st.balloons()



if __name__ == "__main__":
    main()