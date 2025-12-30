"""
Hair type classifier using a trained CNN model.
"""
import cv2
import numpy as np
import os
from pathlib import Path


class HairClassifier:
    """Classify hair type using trained CNN model."""
    
    # Hair type categories (matching training data - alphabetical order)
    STRAIGHT = "Straight"
    WAVY = "Wavy"
    CURLY = "Curly"
    KINKY = "Kinky"
    DREADLOCKS = "Dreadlocks"
    
    # Order must match class_mapping.txt from training
    CLASSES = [STRAIGHT, WAVY, CURLY, DREADLOCKS, KINKY]
    
    def __init__(self, model_path=None):
        """Initialize hair classifier."""
        self.model = None
        
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.parent.resolve()
        models_dir = script_dir / "models"
        
        # Try multiple model formats
        model_paths = [
            models_dir / "hair_type_model.keras",
            models_dir / "hair_type_model.h5",
        ]
        
        if model_path:
            model_paths.insert(0, Path(model_path))
        
        for mp in model_paths:
            if mp.exists():
                self.model_path = mp
                print(f"Found model at: {mp}")
                self._load_model()
                break
        else:
            print(f"No model found. Checked: {[str(p) for p in model_paths]}")
            self.model_path = model_paths[0]
    
    def _load_model(self):
        """Load the trained model."""
        try:
            # Suppress TensorFlow warnings
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            from tensorflow import keras
            self.model = keras.models.load_model(str(self.model_path))
            print(f"Hair classifier model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def is_loaded(self):
        """Check if model is loaded."""
        return self.model is not None
    
    def preprocess(self, image):
        """Preprocess image for model input."""
        # Resize to model input size
        img = cv2.resize(image, (224, 224))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def classify(self, hair_region):
        """Classify hair type from hair region image."""
        if self.model is None:
            return "Unknown", 0.0, {}
        
        if hair_region is None or hair_region.size == 0:
            return "Unknown", 0.0, {}
        
        try:
            # Preprocess image
            img = self.preprocess(hair_region)
            
            # Predict
            predictions = self.model.predict(img, verbose=0)[0]
            
            # Get class with highest probability
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx]) * 100
            
            # Create predictions dict
            all_predictions = {
                self.CLASSES[i]: float(predictions[i]) * 100 
                for i in range(len(self.CLASSES))
            }
            
            hair_type = self.CLASSES[class_idx]
            
            return hair_type, confidence, all_predictions
        except Exception as e:
            print(f"Error classifying: {e}")
            return "Unknown", 0.0, {}
    
    def get_description(self, hair_type):
        """Get description for a hair type."""
        descriptions = {
            self.STRAIGHT: "Rambut lurus, mudah diatur, cocok untuk gaya sleek",
            self.WAVY: "Rambut bergelombang, tekstur natural, versatile",
            self.CURLY: "Rambut keriting, volume natural, butuh hidrasi",
            self.KINKY: "Rambut sangat keriting, tekstur coil, butuh perawatan khusus",
            self.DREADLOCKS: "Gaya dreadlocks, statement style"
        }
        return descriptions.get(hair_type, "")
