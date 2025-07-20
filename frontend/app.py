import streamlit as st 
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image, ImageEnhance

# Set page configuration
st.set_page_config(
    page_title="Deepfake Detector", 
    page_icon="🕵️‍♂️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stFileUploader>div>div>div>button {
            background-color: #4CAF50;
            color: white;
        }
        .prediction-box {
            border-radius: 10px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .real {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
        }
        .fake {
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
        }
        .sidebar .sidebar-content {
            background-color: #343a40;
            color: white;
        }
        .title-text {
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 300;
            letter-spacing: 1px;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model (with error handling)
@st.cache_resource
def load_ai_model():
    try:
        model = load_model('deepfake_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_ai_model()

# Sidebar with information
with st.sidebar:
    st.markdown("""
        <h2 class='title-text'>About</h2>
        <p>This AI-powered tool helps detect deepfake images by analyzing facial features and artifacts.</p>
        <hr>
        <h3 class='title-text'>How it works</h3>
        <ol>
            <li>Upload a clear face image</li>
            <li>AI analyzes facial patterns</li>
            <li>Get instant results</li>
        </ol>
        <hr>
        <p><small>Note: This is a demonstration model. Results may vary based on image quality.</small></p>
    """, unsafe_allow_html=True)

# Main content area
st.markdown("""
    <div style="text-align:center; margin-bottom: 2rem;">
        <h1 class='title-text'>🕵️‍♂️ Deepfake Image Detector</h1>
        <p style="color: #6c757d;">Upload a face image to check for AI manipulation</p>
    </div>
""", unsafe_allow_html=True)

# Function to preprocess and predict with enhanced image processing
def predict_image(image):
    try:
        # Convert to numpy array
        image = np.array(image)
        
        # Enhance image quality before processing
        pil_img = Image.fromarray(image)
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.5)
        
        # Convert back to array and process
        image = np.array(pil_img)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (224, 224))
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # Normalize and prepare for model
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        
        predictions = model.predict(image)
        return predictions
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# File uploader section
upload_col, result_col = st.columns([1, 1])

with upload_col:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a face image...", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(
                img, 
                caption="Uploaded Image", 
                use_container_width=True,
                output_format="PNG"
            )
            
            st.markdown(f"""
                <div style="margin-top: 1rem; padding: 1rem; background-color: #e9ecef; border-radius: 5px;">
                    <p><strong>Image Details:</strong></p>
                    <p>Format: {img.format}</p>
                    <p>Size: {img.size[0]} × {img.size[1]} pixels</p>
                    <p>Mode: {img.mode}</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

# Results section
with result_col:
    st.markdown("### 🔍 Detection Results")
    
    if uploaded_file is not None and model is not None:
        with st.spinner('Analyzing facial patterns...'):
            try:
                predictions = predict_image(img)
                
                if predictions is not None:
                    confidence = float(np.max(predictions)) * 100
                    label = "Real" if np.argmax(predictions) == 0 else "Fake"
                    
                    st.markdown(f"""
                        <div class="prediction-box {'real' if label == 'Real' else 'fake'}">
                            <h2 style="margin-top: 0;">{'✅ Authentic' if label == 'Real' else '❌ Potential Deepfake'}</h2>
                            <div style="margin: 1.5rem 0;">
                                <div style="background: {'#28a745' if label == 'Real' else '#dc3545'}; 
                                    height: 10px; 
                                    border-radius: 5px;
                                    width: {confidence}%;"></div>
                                <p style="text-align: center; margin-top: 5px;">
                                    <strong>{confidence:.2f}% confidence</strong>
                                </p>
                            </div>
                            <p style="font-size: 0.9rem;">
                                {'This image appears to be authentic with no signs of AI manipulation.' 
                                if label == 'Real' 
                                else 'This image shows potential signs of AI manipulation or deepfake generation.'}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("ℹ️ What does this mean?"):
                        if label == "Real":
                            st.success("""
                                Our analysis suggests this is likely a genuine photograph. However, 
                                sophisticated deepfakes can sometimes evade detection. Always verify 
                                important images through multiple sources.
                            """)
                        else:
                            st.warning("""
                                Our analysis detected potential signs of digital manipulation. This could indicate:
                                - AI-generated face (deepfake)
                                - Digital alterations to facial features
                                - Synthetic image artifacts
                                
                                Note that false positives can occur with low-quality images or unusual lighting.
                            """)
            except Exception as e:
                st.error(f"Error during analysis: {e}")
    elif uploaded_file is None:
        st.info("Please upload an image to begin analysis")
        st.image(
            "https://via.placeholder.com/500x300?text=Upload+Image+to+Analyze",
            use_container_width=True
        )

# Footer
st.markdown("""
    <hr>
    <div style="text-align: center; color: #6c757d; font-size: 0.8rem;">
        <p>Deepfake Detector v1.0 | This tool is for educational purposes only</p>
    </div>
""", unsafe_allow_html=True)
