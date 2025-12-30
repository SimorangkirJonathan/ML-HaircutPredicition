"""
Improved Face shape classifier using multiple facial ratios and landmarks.
"""
import numpy as np


class FaceShapeClassifier:
    """Classify face shape based on facial measurements with improved accuracy."""
    
    # Face shape categories
    OVAL = "Oval"
    ROUND = "Round"
    SQUARE = "Square"
    HEART = "Heart"
    OBLONG = "Oblong"
    
    def classify(self, measurements):
        """Classify face shape based on measurements.
        
        Uses multiple ratios for more accurate classification:
        - Length to Width ratio: how elongated the face is
        - Forehead to Jaw ratio: wider forehead = heart, wider jaw = square
        - Jaw to Face ratio: how prominent the jaw is
        """
        ratio = measurements.get('length_to_width_ratio', 1.0)
        forehead_jaw = measurements.get('forehead_to_jaw_ratio', 1.0)
        jaw_face = measurements.get('jaw_to_face_ratio', 0.7)
        
        # Initialize scores
        scores = {
            self.OVAL: 0,
            self.ROUND: 0,
            self.SQUARE: 0,
            self.HEART: 0,
            self.OBLONG: 0
        }
        
        # === OBLONG: Very long face (ratio > 1.5) ===
        if ratio > 1.55:
            scores[self.OBLONG] += 50
        elif ratio > 1.4:
            scores[self.OBLONG] += 30
        elif ratio > 1.3:
            scores[self.OBLONG] += 15
        
        # === ROUND: Short face, wide (ratio < 1.2) ===
        if ratio < 1.15:
            scores[self.ROUND] += 50
        elif ratio < 1.25:
            scores[self.ROUND] += 35
        elif ratio < 1.35:
            scores[self.ROUND] += 20
        
        # === OVAL: Balanced proportions (ratio 1.3-1.5) ===
        if 1.25 <= ratio <= 1.55:
            oval_boost = 40 - abs(ratio - 1.4) * 50  # Peak at 1.4
            scores[self.OVAL] += max(0, oval_boost)
        
        # === HEART: Wide forehead, narrow jaw ===
        if forehead_jaw > 1.25:
            scores[self.HEART] += 45
        elif forehead_jaw > 1.15:
            scores[self.HEART] += 30
        elif forehead_jaw > 1.05:
            scores[self.HEART] += 15
        
        # === SQUARE: Strong jaw, similar width throughout ===
        if jaw_face > 0.82:
            scores[self.SQUARE] += 40
        elif jaw_face > 0.75:
            scores[self.SQUARE] += 25
        
        if 0.95 < forehead_jaw < 1.1 and jaw_face > 0.7:
            scores[self.SQUARE] += 20
        
        # === Additional modifiers ===
        
        # Oval has balanced forehead/jaw
        if 0.95 < forehead_jaw < 1.15:
            scores[self.OVAL] += 15
        
        # Round has soft features
        if forehead_jaw < 1.1 and ratio < 1.3:
            scores[self.ROUND] += 15
        
        # Oblong usually has balanced width
        if ratio > 1.4 and 0.9 < forehead_jaw < 1.15:
            scores[self.OBLONG] += 15
        
        # Heart adjustment
        if forehead_jaw > 1.2 and ratio > 1.2:
            scores[self.HEART] += 10
        
        # Find winner
        max_shape = max(scores, key=scores.get)
        max_score = scores[max_shape]
        
        # Normalize to 0-100
        total = sum(scores.values())
        if total > 0:
            confidence = (max_score / total) * 100
        else:
            confidence = 50
        
        # Minimum confidence boost if score is high
        if max_score >= 50:
            confidence = max(confidence, 70)
        elif max_score >= 40:
            confidence = max(confidence, 60)
        
        details = {
            'length_width_ratio': round(ratio, 3),
            'forehead_jaw_ratio': round(forehead_jaw, 3),
            'jaw_face_ratio': round(jaw_face, 3),
            'all_scores': scores
        }
        
        return max_shape, confidence, details
    
    def get_description(self, face_shape):
        """Get description for a face shape in Indonesian."""
        descriptions = {
            self.OVAL: "Wajah oval - proporsi seimbang, cocok untuk hampir semua gaya rambut",
            self.ROUND: "Wajah bulat - pipi penuh, butuh gaya yang menambah panjang visual",
            self.SQUARE: "Wajah kotak - rahang tegas, butuh gaya yang melunakkan sudut",
            self.HEART: "Wajah hati - dahi lebar dagu runcing, butuh keseimbangan",
            self.OBLONG: "Wajah panjang - butuh gaya yang menambah lebar visual"
        }
        return descriptions.get(face_shape, "")
