"""
Improved Face shape classifier with Diamond shape and better detection.
"""
import numpy as np


class FaceShapeClassifier:
    """Classify face shape based on facial measurements."""
    
    # Face shape categories - NOW INCLUDING DIAMOND
    OVAL = "Oval"
    ROUND = "Round"
    SQUARE = "Square"
    HEART = "Heart"
    OBLONG = "Oblong"
    DIAMOND = "Diamond"  # NEW: Added Diamond face shape
    
    def classify(self, measurements):
        """Classify face shape based on measurements.
        
        Face shapes and their characteristics:
        - OVAL: Length 1.5x width, balanced proportions
        - ROUND: Length ≈ width, soft features
        - SQUARE: Strong jaw, angular, width ≈ length
        - HEART: Wide forehead, narrow chin
        - OBLONG: Very long face, narrow
        - DIAMOND: Wide cheekbones, narrow forehead AND narrow jaw
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
            self.OBLONG: 0,
            self.DIAMOND: 0
        }
        
        # ============================================
        # DIAMOND: Wide cheekbones, narrow forehead AND narrow jaw
        # Cheekbones are the widest part of face
        # Both forehead and jaw are narrower than cheekbones
        # ============================================
        if jaw_face < 0.75:  # Narrow jaw compared to face width
            scores[self.DIAMOND] += 25
        if jaw_face < 0.65:
            scores[self.DIAMOND] += 20
        
        # Diamond has somewhat balanced forehead/jaw (both are narrow)
        if 0.85 < forehead_jaw < 1.15:  # Forehead ≈ Jaw width
            if jaw_face < 0.75:  # But both narrow compared to cheeks
                scores[self.DIAMOND] += 30
        
        # Diamond usually has longer face
        if 1.3 < ratio < 1.6:
            scores[self.DIAMOND] += 15
        
        # ============================================
        # HEART: Wide forehead, narrow jaw/chin
        # ============================================
        if forehead_jaw > 1.25:
            scores[self.HEART] += 45
        elif forehead_jaw > 1.15:
            scores[self.HEART] += 30
        elif forehead_jaw > 1.08:
            scores[self.HEART] += 15
        
        # ============================================
        # OBLONG: Very long face (ratio > 1.5)
        # ============================================
        if ratio > 1.6:
            scores[self.OBLONG] += 50
        elif ratio > 1.5:
            scores[self.OBLONG] += 35
        elif ratio > 1.4:
            scores[self.OBLONG] += 20
        
        # ============================================
        # ROUND: Short face, close to 1:1 ratio
        # ============================================
        if ratio < 1.15:
            scores[self.ROUND] += 50
        elif ratio < 1.25:
            scores[self.ROUND] += 35
        elif ratio < 1.35:
            scores[self.ROUND] += 20
        
        # Round has soft jaw
        if jaw_face > 0.75 and ratio < 1.3:
            scores[self.ROUND] += 15
        
        # ============================================
        # SQUARE: Strong angular jaw, width ≈ length
        # ============================================
        if jaw_face > 0.85:
            scores[self.SQUARE] += 40
        elif jaw_face > 0.78:
            scores[self.SQUARE] += 25
        
        if 0.95 < forehead_jaw < 1.1 and jaw_face > 0.75:
            scores[self.SQUARE] += 20
        
        if ratio < 1.3 and jaw_face > 0.8:
            scores[self.SQUARE] += 15
        
        # ============================================
        # OVAL: Balanced proportions (ratio 1.3-1.5)
        # ============================================
        if 1.25 <= ratio <= 1.55:
            oval_boost = 35 - abs(ratio - 1.4) * 40  # Peak at 1.4
            scores[self.OVAL] += max(0, oval_boost)
        
        # Oval has balanced forehead/jaw
        if 0.95 < forehead_jaw < 1.15:
            scores[self.OVAL] += 15
        
        # Oval has moderate jaw
        if 0.7 < jaw_face < 0.85:
            scores[self.OVAL] += 10
        
        # ============================================
        # Find winner
        # ============================================
        max_shape = max(scores, key=scores.get)
        max_score = scores[max_shape]
        
        # Normalize to 0-100
        total = sum(scores.values())
        if total > 0:
            confidence = (max_score / total) * 100
        else:
            confidence = 50
        
        # Boost confidence if score is high
        if max_score >= 50:
            confidence = max(confidence, 75)
        elif max_score >= 40:
            confidence = max(confidence, 65)
        elif max_score >= 30:
            confidence = max(confidence, 55)
        
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
            self.OBLONG: "Wajah panjang - butuh gaya yang menambah lebar visual",
            self.DIAMOND: "Wajah diamond - tulang pipi lebar, dahi dan dagu sempit"
        }
        return descriptions.get(face_shape, "")
