"""
Hairstyle Recommendation App - Streamlit Web Interface

Modern web-based UI for real-time face shape and hair type detection
with hairstyle recommendations.

Run with: streamlit run app_streamlit.py
"""
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
import os
import av
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.face_detector import FaceDetector
from src.face_shape_classifier import FaceShapeClassifier
from src.hair_classifier import HairClassifier
from src.recommender import HairstyleRecommender


# ============== Page Configuration ==============
st.set_page_config(
    page_title="Hairstyle Recommendation",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== Custom CSS Styling ==============
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #f1f5f9;
    }
    
    /* Card styling */
    .result-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid #475569;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .result-card h3 {
        color: #6366f1;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .result-card .value {
        color: #10b981;
        font-size: 1.8rem;
        font-weight: bold;
    }
    
    .result-card .confidence {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    /* Recommendation card */
    .recommendation-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #475569;
    }
    
    .style-item {
        background: rgba(99, 102, 241, 0.1);
        border-left: 3px solid #6366f1;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        color: #f1f5f9;
    }
    
    .tips-box {
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid #10b981;
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 0 8px 8px 0;
        color: #f1f5f9;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Status indicator */
    .status-active {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    
    .status-inactive {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #475569;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #10b981 !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
</style>
""", unsafe_allow_html=True)


# ============== Initialize Components (Cached) ==============
@st.cache_resource
def load_face_detector():
    return FaceDetector()

@st.cache_resource
def load_face_shape_classifier():
    return FaceShapeClassifier()

@st.cache_resource
def load_hair_classifier():
    return HairClassifier()

@st.cache_resource
def load_recommender():
    return HairstyleRecommender()


# ============== Shared State for Results ==============
class ResultState:
    """Thread-safe state for sharing results between video processor and UI."""
    def __init__(self):
        self.lock = threading.Lock()
        self.face_shape = None
        self.face_confidence = 0
        self.hair_type = None
        self.hair_confidence = 0
        self.recommendations = None
        self.face_detected = False
        self.has_real_landmarks = False
    
    def update(self, face_shape, face_conf, hair_type, hair_conf, recommendations, face_detected, real_landmarks):
        with self.lock:
            self.face_shape = face_shape
            self.face_confidence = face_conf
            self.hair_type = hair_type
            self.hair_confidence = hair_conf
            self.recommendations = recommendations
            self.face_detected = face_detected
            self.has_real_landmarks = real_landmarks
    
    def get(self):
        with self.lock:
            return {
                'face_shape': self.face_shape,
                'face_confidence': self.face_confidence,
                'hair_type': self.hair_type,
                'hair_confidence': self.hair_confidence,
                'recommendations': self.recommendations,
                'face_detected': self.face_detected,
                'has_real_landmarks': self.has_real_landmarks
            }


# Module-level global shared state (accessible from any thread)
RESULT_STATE = ResultState()


# ============== Video Processor for WebRTC ==============
class HairstyleVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_detector = load_face_detector()
        self.face_shape_classifier = load_face_shape_classifier()
        self.hair_classifier = load_hair_classifier()
        self.recommender = load_recommender()
        
        self.show_landmarks = True
        
        # Local state for smoothing
        self.face_shape = None
        self.face_confidence = 0
        self.hair_type = None
        self.hair_confidence = 0
        self.recommendations = None
        
        # Temporal smoothing
        self.SMOOTHING_WINDOW = 15
        self.face_shape_history = []
        self.hair_type_history = []
        self.face_confidence_history = []
        self.hair_confidence_history = []
        
        # Colors (BGR for OpenCV)
        self.COLOR_PRIMARY = (241, 102, 99)  # Indigo
        self.COLOR_SUCCESS = (129, 185, 16)  # Green
        self.COLOR_ACCENT = (212, 182, 6)    # Cyan
    
    def get_smoothed_prediction(self, history, new_value, confidence_history=None, new_conf=0):
        """Get smoothed prediction from history using voting."""
        if new_value is not None and new_value != "Unknown":
            history.append(new_value)
            if confidence_history is not None:
                confidence_history.append(new_conf)
        
        if len(history) > self.SMOOTHING_WINDOW:
            history.pop(0)
            if confidence_history is not None:
                confidence_history.pop(0)
        
        if not history:
            return None, 0
        
        from collections import Counter
        counts = Counter(history)
        most_common = counts.most_common(1)[0]
        smoothed_value = most_common[0]
        vote_ratio = most_common[1] / len(history)
        
        if confidence_history:
            avg_conf = sum(confidence_history) / len(confidence_history)
        else:
            avg_conf = vote_ratio * 100
        
        return smoothed_value, avg_conf
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detect faces
        faces = self.face_detector.detect_faces(img)
        face_detected = False
        real_landmarks = False
        
        if len(faces) > 0:
            face_detected = True
            face_rect = faces[0]
            landmarks = self.face_detector.get_landmarks(img, face_rect)
            real_landmarks = landmarks.get('real_landmarks', False)
            
            # Draw face rectangle
            x, y, w, h = landmarks['bbox']
            cv2.rectangle(img, (x, y), (x + w, y + h), self.COLOR_PRIMARY, 2)
            
            # Draw landmarks if enabled
            if self.show_landmarks:
                self.face_detector.draw_landmarks(img, landmarks)
                self.face_detector.draw_hair_region(img, self.COLOR_ACCENT)
            
            # Classify face shape
            measurements = self.face_detector.get_face_measurements(landmarks)
            raw_face_shape, raw_face_conf, _ = self.face_shape_classifier.classify(measurements)
            
            self.face_shape, self.face_confidence = self.get_smoothed_prediction(
                self.face_shape_history, raw_face_shape,
                self.face_confidence_history, raw_face_conf
            )
            
            # Classify hair type
            hair_region = self.face_detector.get_hair_region(img, landmarks)
            if hair_region is not None and hair_region.size > 0:
                raw_hair_type, raw_hair_conf, _ = self.hair_classifier.classify(hair_region)
                
                self.hair_type, self.hair_confidence = self.get_smoothed_prediction(
                    self.hair_type_history, raw_hair_type,
                    self.hair_confidence_history, raw_hair_conf
                )
            
            # Get recommendations
            if self.face_shape and self.hair_type and self.hair_type != "Unknown":
                self.recommendations = self.recommender.get_recommendation(self.face_shape, self.hair_type)
            
            # Draw status  
            cv2.putText(img, "FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_SUCCESS, 2)
            if real_landmarks:
                cv2.putText(img, "68 LANDMARKS", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_ACCENT, 1)
        else:
            cv2.putText(img, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
            cv2.putText(img, "Position your face in center", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Update shared state
        RESULT_STATE.update(
            self.face_shape, self.face_confidence,
            self.hair_type, self.hair_confidence,
            self.recommendations, face_detected, real_landmarks
        )
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============== Main App ==============
def main():
    # Sidebar
    with st.sidebar:
        st.markdown('<p class="main-title">✂️ Hairstyle AI</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Smart Recommendation System</p>', unsafe_allow_html=True)
        
        st.divider()
        
        # Settings
        st.markdown("### ⚙️ Settings")
        show_landmarks = st.toggle("Show Facial Landmarks", value=True)
        auto_refresh = st.toggle("Auto Refresh Results", value=True, help="Otomatis update hasil setiap 1 detik")
        
        st.divider()
        
        # Info
        st.markdown("### 📊 Model Info")
        st.markdown("""
        - **Face Detection**: Haar Cascade
        - **Landmarks**: 68-point LBF
        - **Hair Model**: MobileNetV2
        - **Accuracy**: ~85%
        """)
        
        st.divider()
        
        # Controls info
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Pastikan pencahayaan cukup
        - Posisikan wajah di tengah
        - Jaga jarak ~50cm dari kamera
        """)
    
    # Main content
    st.markdown('<p class="main-title">Hairstyle Recommendation System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time AI-powered face shape and hair type analysis</p>', unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Camera")
        
        # WebRTC Streamer
        ctx = webrtc_streamer(
            key="hairstyle-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=HairstyleVideoProcessor,
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
        
        # Update landmarks toggle
        if ctx.video_processor:
            ctx.video_processor.show_landmarks = show_landmarks
    
    with col2:
        st.markdown("### 📊 Detection Results")
        
        # Get results from shared state
        results = RESULT_STATE.get()
        
        # Status indicator
        if results['face_detected']:
            status_text = "🟢 Face Detected"
            if results['has_real_landmarks']:
                status_text += " (68 Landmarks)"
            st.markdown(f'<span class="status-active">{status_text}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-inactive">⚪ Waiting for face...</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Face Shape Result
        st.markdown("""
        <div class="result-card">
            <h3>🎭 Face Shape</h3>
            <div class="value">{}</div>
            <div class="confidence">Confidence: {:.0f}%</div>
        </div>
        """.format(
            results['face_shape'] or "Detecting...",
            results['face_confidence'] if results['face_shape'] else 0
        ), unsafe_allow_html=True)
        
        if results['face_confidence'] > 0:
            st.progress(min(int(results['face_confidence']), 100) / 100)
        
        # Hair Type Result
        st.markdown("""
        <div class="result-card">
            <h3>💇 Hair Type</h3>
            <div class="value">{}</div>
            <div class="confidence">Confidence: {:.0f}%</div>
        </div>
        """.format(
            results['hair_type'] if results['hair_type'] and results['hair_type'] != "Unknown" else "Detecting...",
            results['hair_confidence'] if results['hair_type'] else 0
        ), unsafe_allow_html=True)
        
        if results['hair_confidence'] > 0:
            st.progress(min(int(results['hair_confidence']), 100) / 100)
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### ✨ Recommended Styles")
        
        if results['recommendations']:
            for i, style in enumerate(results['recommendations'].get('styles', [])[:4], 1):
                st.markdown(f"""
                <div class="style-item">
                    <strong>{i}.</strong> {style}
                </div>
                """, unsafe_allow_html=True)
            
            tips = results['recommendations'].get('tips', '')
            if tips:
                st.markdown(f"""
                <div class="tips-box">
                    <strong>💡 Tips:</strong><br>{tips}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Menunggu deteksi wajah dan rambut...")
        
        # Auto refresh
        if auto_refresh and ctx.state.playing:
            import time
            time.sleep(1)
            st.rerun()


if __name__ == "__main__":
    main()
