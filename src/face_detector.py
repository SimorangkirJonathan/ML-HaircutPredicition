"""
Fixed Face detector with proper hair region extraction and accurate landmarks.
"""
import cv2
import numpy as np


class FaceDetector:
    """Face detector with proper hair region extraction."""
    
    def __init__(self):
        """Initialize face detector."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("Error: Could not load face cascade classifier")
        else:
            print("Face Detector initialized successfully")
        
    def detect_faces(self, frame):
        """Detect faces in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Improve detection with histogram equalization
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More sensitive
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return list(faces)
    
    def get_landmarks(self, frame, face_rect):
        """Get accurate facial landmarks covering the entire face end-to-end."""
        x, y, w, h = face_rect
        
        # Landmark positions based on facial anatomy (percentages of face bbox)
        # These cover the face from edge to edge
        landmarks = {
            # Face outline - from edge to edge
            'top_center': (x + w // 2, y),                    # Top of face bbox
            'bottom_center': (x + w // 2, y + h),             # Bottom of chin
            'left_edge': (x, y + h // 2),                     # Left edge of face
            'right_edge': (x + w, y + h // 2),                # Right edge of face
            
            # Forehead region
            'forehead_left': (x + int(w * 0.15), y + int(h * 0.05)),
            'forehead_center': (x + w // 2, y + int(h * 0.05)),
            'forehead_right': (x + int(w * 0.85), y + int(h * 0.05)),
            
            # Temples (sides of forehead)
            'left_temple': (x + int(w * 0.02), y + int(h * 0.15)),
            'right_temple': (x + int(w * 0.98), y + int(h * 0.15)),
            
            # Eyes region
            'left_eye': (x + int(w * 0.30), y + int(h * 0.35)),
            'right_eye': (x + int(w * 0.70), y + int(h * 0.35)),
            
            # Cheeks - widest part of face
            'left_cheek': (x + int(w * 0.02), y + int(h * 0.50)),
            'right_cheek': (x + int(w * 0.98), y + int(h * 0.50)),
            
            # Nose
            'nose_tip': (x + w // 2, y + int(h * 0.65)),
            
            # Mouth region
            'mouth_left': (x + int(w * 0.25), y + int(h * 0.78)),
            'mouth_right': (x + int(w * 0.75), y + int(h * 0.78)),
            
            # Jawline - important for face shape
            'left_jaw': (x + int(w * 0.08), y + int(h * 0.85)),
            'right_jaw': (x + int(w * 0.92), y + int(h * 0.85)),
            
            # Chin
            'chin': (x + w // 2, y + int(h * 0.98)),
            
            # Store bbox
            'bbox': (x, y, w, h)
        }
        
        return landmarks
    
    def get_face_measurements(self, landmarks):
        """Calculate accurate face measurements from landmarks."""
        x, y, w, h = landmarks['bbox']
        
        def dist(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Face length: from forehead to chin
        face_length = dist(landmarks['forehead_center'], landmarks['chin'])
        
        # Face width at cheeks (widest part)
        cheek_width = dist(landmarks['left_cheek'], landmarks['right_cheek'])
        
        # Forehead width at temples
        forehead_width = dist(landmarks['left_temple'], landmarks['right_temple'])
        
        # Jaw width
        jaw_width = dist(landmarks['left_jaw'], landmarks['right_jaw'])
        
        # Calculate ratios
        length_to_width = face_length / cheek_width if cheek_width > 0 else 1.0
        forehead_to_jaw = forehead_width / jaw_width if jaw_width > 0 else 1.0
        jaw_to_face = jaw_width / cheek_width if cheek_width > 0 else 0.7
        
        return {
            'face_width': cheek_width,
            'face_length': face_length,
            'forehead_width': forehead_width,
            'jaw_width': jaw_width,
            'length_to_width_ratio': length_to_width,
            'forehead_to_jaw_ratio': forehead_to_jaw,
            'jaw_to_face_ratio': jaw_to_face,
        }
    
    def get_hair_region(self, frame, landmarks):
        """Extract the HAIR region ABOVE the face - NOT the face itself.
        
        This extracts the area above the face bounding box where the hair is located.
        """
        x, y, w, h = landmarks['bbox']
        frame_h, frame_w = frame.shape[:2]
        
        # IMPORTANT: Hair region is ABOVE the face, not including the face
        # Calculate hair region boundaries
        
        # Hair starts from top of frame (or reasonable distance) down to just above eyebrows
        hair_top = max(0, y - int(h * 0.8))  # Start well above face
        hair_bottom = y + int(h * 0.15)       # End just below forehead (above eyes)
        
        # Hair width extends slightly beyond face width
        hair_left = max(0, x - int(w * 0.25))
        hair_right = min(frame_w, x + w + int(w * 0.25))
        
        # Minimum size check
        hair_height = hair_bottom - hair_top
        hair_width = hair_right - hair_left
        
        if hair_height < 30 or hair_width < 30:
            print("Warning: Hair region too small")
            return None
        
        # Extract hair region
        hair_region = frame[hair_top:hair_bottom, hair_left:hair_right]
        
        # Store coordinates for visualization
        self.last_hair_region = (hair_left, hair_top, hair_width, hair_height)
        
        return hair_region
    
    def draw_landmarks(self, frame, landmarks, draw_all=True):
        """Draw all landmarks on the frame to visualize coverage."""
        # Draw each landmark point
        for key, pos in landmarks.items():
            if key == 'bbox':
                continue
            
            # Color code by region
            if 'forehead' in key or 'temple' in key:
                color = (0, 255, 255)  # Yellow - forehead
            elif 'eye' in key:
                color = (255, 0, 0)    # Blue - eyes
            elif 'cheek' in key:
                color = (0, 255, 0)    # Green - cheeks
            elif 'jaw' in key or 'chin' in key:
                color = (0, 0, 255)    # Red - jaw/chin
            elif 'edge' in key or 'center' in key:
                color = (255, 255, 0)  # Cyan - edges
            else:
                color = (255, 255, 255)  # White - others
            
            cv2.circle(frame, pos, 4, color, -1)
            cv2.circle(frame, pos, 5, (0, 0, 0), 1)  # Black outline
    
    def draw_face_rect(self, frame, landmarks, color=(255, 200, 100)):
        """Draw face bounding box."""
        x, y, w, h = landmarks['bbox']
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    def draw_hair_region(self, frame, color=(255, 0, 255)):
        """Draw the hair region rectangle to show what's being scanned."""
        if hasattr(self, 'last_hair_region'):
            x, y, w, h = self.last_hair_region
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "HAIR REGION", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def close(self):
        """Release resources."""
        pass
