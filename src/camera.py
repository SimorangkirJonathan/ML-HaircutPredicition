"""
Camera module for capturing video frames from webcam.
"""
import cv2


class Camera:
    def __init__(self, camera_id=0, width=640, height=480):
        """Initialize camera capture.
        
        Args:
            camera_id: Camera device ID (default 0 for primary webcam)
            width: Frame width
            height: Frame height
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height
        
    def read(self):
        """Read a frame from the camera.
        
        Returns:
            tuple: (success, frame) where success is bool and frame is numpy array
        """
        ret, frame = self.cap.read()
        if ret:
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
        return ret, frame
    
    def release(self):
        """Release the camera resource."""
        self.cap.release()
        
    def is_opened(self):
        """Check if camera is opened."""
        return self.cap.isOpened()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
