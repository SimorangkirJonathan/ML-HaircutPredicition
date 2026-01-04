"""
Hairstyle Recommendation App - Main Application

Real-time face shape and hair type detection with hairstyle recommendations.

Controls:
    Q - Quit
    S - Screenshot
    L - Toggle landmarks display
    H - Toggle help overlay
"""
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.camera import Camera
from src.face_detector import FaceDetector
from src.face_shape_classifier import FaceShapeClassifier
from src.hair_classifier import HairClassifier
from src.recommender import HairstyleRecommender


class HairstyleApp:
    """Main application class."""
    
    def __init__(self):
        """Initialize the application."""
        print("Initializing Hairstyle Recommendation App...")
        
        self.camera = Camera()
        self.face_detector = FaceDetector()
        self.face_shape_classifier = FaceShapeClassifier()
        self.hair_classifier = HairClassifier()
        self.recommender = HairstyleRecommender()
        
        self.show_landmarks = True
        self.show_help = True
        
        # Colors (BGR)
        self.COLOR_PRIMARY = (255, 200, 100)  # Light blue
        self.COLOR_SUCCESS = (100, 255, 100)  # Green
        self.COLOR_WARNING = (100, 200, 255)  # Orange
        self.COLOR_TEXT = (255, 255, 255)     # White
        self.COLOR_BG = (50, 50, 50)          # Dark gray
        
    def draw_text_with_bg(self, frame, text, pos, font_scale=0.6, color=None, bg_color=None):
        """Draw text with background for better visibility."""
        if color is None:
            color = self.COLOR_TEXT
        if bg_color is None:
            bg_color = self.COLOR_BG
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        x, y = pos
        padding = 5
        cv2.rectangle(
            frame,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(frame, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)
        
        return text_height + baseline + padding * 2
    
    def draw_panel(self, frame, x, y, width, height, title=""):
        """Draw a semi-transparent panel."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.COLOR_PRIMARY, 2)
        
        if title:
            self.draw_text_with_bg(frame, title, (x + 10, y + 25), 0.7, self.COLOR_PRIMARY)
    
    def draw_info_panel(self, frame, face_shape, face_confidence, hair_type, hair_confidence, recommendations):
        """Draw a compact information panel with results."""
        h, w = frame.shape[:2]
        panel_width = 220  # Smaller width
        panel_x = w - panel_width - 5
        panel_y = 5
        
        # Calculate dynamic height based on content
        panel_height = 200  # Compact height
        if recommendations and 'styles' in recommendations:
            panel_height += len(recommendations['styles'][:3]) * 18 + 40
        
        # Draw panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), self.COLOR_PRIMARY, 1)
        
        y_offset = panel_y + 18
        
        # Title - smaller
        cv2.putText(frame, "RESULTS", (panel_x + 8, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_PRIMARY, 1, cv2.LINE_AA)
        y_offset += 22
        
        # Face Shape - compact
        cv2.putText(frame, "Face:", (panel_x + 8, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WARNING, 1, cv2.LINE_AA)
        if face_shape:
            cv2.putText(frame, f"{face_shape} {face_confidence:.0f}%", (panel_x + 50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_SUCCESS, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "---", (panel_x + 50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        y_offset += 20
        
        # Hair Type - compact
        cv2.putText(frame, "Hair:", (panel_x + 8, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WARNING, 1, cv2.LINE_AA)
        if hair_type and hair_type != "Unknown":
            cv2.putText(frame, f"{hair_type} {hair_confidence:.0f}%", (panel_x + 50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_SUCCESS, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "---", (panel_x + 50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        y_offset += 18
        
        # Divider
        cv2.line(frame, (panel_x + 5, y_offset), (panel_x + panel_width - 5, y_offset), self.COLOR_PRIMARY, 1)
        y_offset += 15
        
        # Recommendations - compact
        cv2.putText(frame, "Styles:", (panel_x + 8, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_PRIMARY, 1, cv2.LINE_AA)
        y_offset += 18
        
        if recommendations and 'styles' in recommendations:
            for i, style in enumerate(recommendations['styles'][:3]):  # Max 3 styles
                cv2.putText(frame, f"{i+1}. {style}", (panel_x + 8, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLOR_TEXT, 1, cv2.LINE_AA)
                y_offset += 16
    
    def draw_help(self, frame):
        """Draw help overlay."""
        if not self.show_help:
            return
            
        h, w = frame.shape[:2]
        y_offset = h - 40
        
        help_text = "Q: Quit | S: Screenshot | L: Toggle Landmarks | H: Toggle Help"
        self.draw_text_with_bg(frame, help_text, (10, y_offset), 0.5, (200, 200, 200))
    
    def draw_model_status(self, frame):
        """Draw model loading status."""
        if not self.hair_classifier.is_loaded():
            self.draw_text_with_bg(
                frame, 
                "! Hair model not loaded. Run: python train_hair_model.py", 
                (10, 30), 
                0.6, 
                (100, 100, 255),
                (50, 50, 100)
            )
    
    def run(self):
        """Run the main application loop."""
        print("=" * 50)
        print("HAIRSTYLE RECOMMENDATION APP")
        print("=" * 50)
        print("\nControls:")
        print("  Q - Quit")
        print("  S - Screenshot")
        print("  L - Toggle landmarks display")
        print("  H - Toggle help overlay")
        print("\nStarting camera...")
        
        if not self.camera.is_opened():
            print("Error: Could not open camera!")
            return
        
        print("Camera opened successfully!")
        print("Press Q to quit.\n")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Could not read frame!")
                break
            
            # Initialize results
            face_shape = None
            face_confidence = 0
            hair_type = None
            hair_confidence = 0
            recommendations = None
            
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            if len(faces) > 0:
                face_rect = faces[0]  # Use first detected face
                
                # Get landmarks
                landmarks = self.face_detector.get_landmarks(frame, face_rect)
                
                # Draw face rectangle
                self.face_detector.draw_face_rect(frame, landmarks, self.COLOR_PRIMARY)
                
                # Draw landmarks
                if self.show_landmarks:
                    self.face_detector.draw_landmarks(frame, landmarks)
                    # Also draw hair region to show what's being scanned
                    self.face_detector.draw_hair_region(frame, (255, 0, 255))  # Magenta
                
                # Classify face shape
                measurements = self.face_detector.get_face_measurements(landmarks)
                face_shape, face_confidence, details = self.face_shape_classifier.classify(measurements)
                
                # DEBUG: Print measurements to console
                print(f"L/W: {details['length_width_ratio']:.2f}, FH/J: {details['forehead_jaw_ratio']:.2f}, J/F: {details['jaw_face_ratio']:.2f} -> {face_shape}")
                
                # Get hair region and classify
                hair_region = self.face_detector.get_hair_region(frame, landmarks)
                if hair_region is not None and hair_region.size > 0:
                    hair_type, hair_confidence, _ = self.hair_classifier.classify(hair_region)
                    # Draw hair region after getting it
                    self.face_detector.draw_hair_region(frame, (255, 0, 255))
                
                # Get recommendations
                if face_shape and hair_type and hair_type != "Unknown":
                    recommendations = self.recommender.get_recommendation(face_shape, hair_type)
                elif face_shape:
                    # Fallback to face shape only recommendations
                    all_recs = self.recommender.get_all_for_face_shape(face_shape)
                    if all_recs:
                        first_hair_type = list(all_recs.keys())[0]
                        recommendations = all_recs[first_hair_type]
            
            # Draw UI elements
            self.draw_info_panel(frame, face_shape, face_confidence, hair_type, hair_confidence, recommendations)
            self.draw_help(frame)
            self.draw_model_status(frame)
            
            # Show frame
            cv2.imshow("Hairstyle Recommendation", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("Quitting...")
                break
            elif key == ord('s') or key == ord('S'):
                filename = "screenshot.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('l') or key == ord('L'):
                self.show_landmarks = not self.show_landmarks
                print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
            elif key == ord('h') or key == ord('H'):
                self.show_help = not self.show_help
                print(f"Help: {'ON' if self.show_help else 'OFF'}")
        
        # Cleanup
        self.camera.release()
        self.face_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = HairstyleApp()
    app.run()
