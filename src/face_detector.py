"""
Face detector with REAL facial landmarks using OpenCV Facemark LBF.
This provides 68 facial landmarks for accurate face shape detection.
"""
import cv2
import numpy as np
import os
from pathlib import Path


class FaceDetector:
    """Face detector with 68 facial landmarks."""
    
    def __init__(self):
        """Initialize face detector with landmark model."""
        # Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("Error: Could not load face cascade classifier")
            return
        
        # Try to load Facemark LBF for 68 landmarks
        self.facemark = None
        self.use_facemark = False
        
        try:
            self.facemark = cv2.face.createFacemarkLBF()
            
            # Model path
            script_dir = Path(__file__).parent.parent.resolve()
            model_path = script_dir / "models" / "lbfmodel.yaml"
            
            if model_path.exists():
                self.facemark.loadModel(str(model_path))
                self.use_facemark = True
                print(f"Facemark LBF loaded - 68 real landmarks enabled!")
            else:
                print(f"LBF model not found at {model_path}")
                print("Using estimated landmarks instead.")
        except Exception as e:
            print(f"Facemark not available: {e}")
            print("Using estimated landmarks instead.")
        
        print("Face Detector initialized successfully")
        
        # 68 landmark indices for face shape analysis
        # Jawline: 0-16
        # Right eyebrow: 17-21
        # Left eyebrow: 22-26
        # Nose: 27-35
        # Right eye: 36-41
        # Left eye: 42-47
        # Mouth: 48-67
        
        self.last_hair_region = None
        
    def detect_faces(self, frame):
        """Detect faces in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return list(faces)
    
    def get_real_landmarks(self, frame, face_rect):
        """Get REAL 68 facial landmarks using Facemark LBF."""
        if not self.use_facemark:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Facemark needs faces as numpy array
        faces_array = np.array([[face_rect[0], face_rect[1], 
                                  face_rect[2], face_rect[3]]], dtype=np.int32)
        
        try:
            success, landmarks = self.facemark.fit(gray, faces_array)
            if success and len(landmarks) > 0:
                return landmarks[0][0]  # First face, first result, shape (68, 2)
        except Exception as e:
            print(f"Landmark detection error: {e}")
        
        return None
    
    def get_landmarks(self, frame, face_rect):
        """Get facial landmarks - real if available, estimated otherwise."""
        x, y, w, h = face_rect
        
        # Try to get real landmarks
        real_landmarks = self.get_real_landmarks(frame, face_rect)
        
        if real_landmarks is not None:
            # Use real 68 landmarks
            pts = real_landmarks.astype(np.int32)
            
            landmarks = {
                # Jawline points
                'jaw_left': tuple(pts[0]),
                'jaw_right': tuple(pts[16]),
                'chin': tuple(pts[8]),
                
                # Cheekbones (widest part - approximate from jawline)
                'left_cheek': tuple(pts[2]),
                'right_cheek': tuple(pts[14]),
                
                # Forehead (estimate from eyebrows)
                'left_temple': (int(pts[17][0]), int(pts[17][1] - h * 0.15)),
                'right_temple': (int(pts[26][0]), int(pts[26][1] - h * 0.15)),
                'forehead_center': (int((pts[17][0] + pts[26][0]) / 2), 
                                   int(pts[21][1] - h * 0.15)),
                
                # Eyes
                'left_eye': tuple(((pts[36] + pts[39]) / 2).astype(int)),
                'right_eye': tuple(((pts[42] + pts[45]) / 2).astype(int)),
                
                # Nose
                'nose_tip': tuple(pts[30]),
                
                # Mouth
                'mouth_left': tuple(pts[48]),
                'mouth_right': tuple(pts[54]),
                
                # All 68 landmarks for drawing
                'all_68': pts,
                
                # Use real landmarks flag
                'real_landmarks': True,
                
                'bbox': (x, y, w, h)
            }
        else:
            # Fallback to estimated landmarks
            landmarks = {
                'jaw_left': (x + int(w * 0.08), y + int(h * 0.85)),
                'jaw_right': (x + int(w * 0.92), y + int(h * 0.85)),
                'chin': (x + w // 2, y + int(h * 0.98)),
                
                'left_cheek': (x + int(w * 0.02), y + int(h * 0.50)),
                'right_cheek': (x + int(w * 0.98), y + int(h * 0.50)),
                
                'left_temple': (x + int(w * 0.02), y + int(h * 0.15)),
                'right_temple': (x + int(w * 0.98), y + int(h * 0.15)),
                'forehead_center': (x + w // 2, y + int(h * 0.05)),
                
                'left_eye': (x + int(w * 0.30), y + int(h * 0.35)),
                'right_eye': (x + int(w * 0.70), y + int(h * 0.35)),
                
                'nose_tip': (x + w // 2, y + int(h * 0.65)),
                
                'mouth_left': (x + int(w * 0.25), y + int(h * 0.78)),
                'mouth_right': (x + int(w * 0.75), y + int(h * 0.78)),
                
                'all_68': None,
                'real_landmarks': False,
                
                'bbox': (x, y, w, h)
            }
        
        return landmarks
    
    def get_face_measurements(self, landmarks):
        """Calculate face measurements from landmarks."""
        def dist(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Calculate key distances
        face_length = dist(landmarks['forehead_center'], landmarks['chin'])
        cheek_width = dist(landmarks['left_cheek'], landmarks['right_cheek'])
        forehead_width = dist(landmarks['left_temple'], landmarks['right_temple'])
        jaw_width = dist(landmarks['jaw_left'], landmarks['jaw_right'])
        
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
            'real_landmarks': landmarks.get('real_landmarks', False)
        }
    
    def get_hair_region(self, frame, landmarks):
        """Extract the hair region above the face."""
        x, y, w, h = landmarks['bbox']
        frame_h, frame_w = frame.shape[:2]
        
        hair_height = int(h * 0.40)
        hair_top = max(0, y - hair_height)
        hair_bottom = y + int(h * 0.12)
        
        hair_left = max(0, x - int(w * 0.10))
        hair_right = min(frame_w, x + w + int(w * 0.10))
        
        actual_height = hair_bottom - hair_top
        actual_width = hair_right - hair_left
        
        if actual_height < 20 or actual_width < 20:
            return None
        
        hair_region = frame[hair_top:hair_bottom, hair_left:hair_right]
        self.last_hair_region = (hair_left, hair_top, actual_width, actual_height)
        
        return hair_region
    
    def draw_landmarks(self, frame, landmarks, draw_all=True):
        """Draw landmarks on the frame."""
        # If we have real 68 landmarks, draw them
        if landmarks.get('all_68') is not None:
            pts = landmarks['all_68']
            
            # Draw jawline (0-16) in red
            for i in range(16):
                cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (0, 0, 255), 1)
            
            # Draw eyebrows in yellow
            for i in range(17, 21):
                cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (0, 255, 255), 1)
            for i in range(22, 26):
                cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (0, 255, 255), 1)
            
            # Draw nose in green
            for i in range(27, 30):
                cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (0, 255, 0), 1)
            for i in range(31, 35):
                cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (0, 255, 0), 1)
            
            # Draw eyes in blue
            for i in range(36, 41):
                cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (255, 0, 0), 1)
            cv2.line(frame, tuple(pts[41]), tuple(pts[36]), (255, 0, 0), 1)
            for i in range(42, 47):
                cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (255, 0, 0), 1)
            cv2.line(frame, tuple(pts[47]), tuple(pts[42]), (255, 0, 0), 1)
            
            # Draw mouth in pink
            for i in range(48, 59):
                cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (255, 0, 255), 1)
            cv2.line(frame, tuple(pts[59]), tuple(pts[48]), (255, 0, 255), 1)
            
            # Draw all points
            for pt in pts:
                cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)
        else:
            # Draw estimated landmarks
            for key, pos in landmarks.items():
                if key in ['bbox', 'all_68', 'real_landmarks']:
                    continue
                if isinstance(pos, tuple) and len(pos) == 2:
                    cv2.circle(frame, pos, 4, (0, 255, 0), -1)
            
    def draw_face_rect(self, frame, landmarks, color=(255, 200, 100)):
        """Draw face bounding box."""
        x, y, w, h = landmarks['bbox']
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Indicate if using real landmarks
        if landmarks.get('real_landmarks', False):
            cv2.putText(frame, "68 LANDMARKS", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def draw_hair_region(self, frame, color=(255, 0, 255)):
        """Draw the hair region rectangle."""
        if self.last_hair_region:
            x, y, w, h = self.last_hair_region
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "HAIR", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def close(self):
        """Release resources."""
        pass
