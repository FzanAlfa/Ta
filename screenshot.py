import cv2
import os
from datetime import datetime

class SecurityLogger:
    def __init__(self):
        """Initialize the security logger"""
        self.base_dir = 'wrongakses'
        self._ensure_base_dir()

    def _ensure_base_dir(self):
        """Ensure the base directory exists"""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _get_date_folder(self):
        """Get the current date folder name and ensure it exists"""
        current_date = datetime.now()
        folder_name = current_date.strftime("%d_%m_%Y")  # DD_MM_YYYY format
        folder_path = os.path.join(self.base_dir, folder_name)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        return folder_path

    def save_unauthorized_attempt(self, frame, confidence=0.0):
        """
        Save an unauthorized access attempt
        Args:
            frame: The captured frame/image
            confidence: The confidence score of the failed match (if any)
        """
        try:
            # Get current date folder
            date_folder = self._get_date_folder()
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%H_%M_%S")  # HH_MM_SS format
            filename = f"unauthorized_{timestamp}_conf_{confidence:.2f}.jpg"
            filepath = os.path.join(date_folder, filename)
            
            # Save the image
            cv2.imwrite(filepath, frame)
            
            print(f"Unauthorized access attempt logged: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving unauthorized attempt: {str(e)}")
            return None

    def list_unauthorized_attempts(self, date=None):
        """
        List unauthorized attempts, optionally filtered by date
        Args:
            date: Optional date string in DD_MM_YYYY format
        Returns:
            List of unauthorized attempt records
        """
        attempts = []
        
        if date:
            # Check specific date folder
            folder_path = os.path.join(self.base_dir, date)
            if os.path.exists(folder_path):
                attempts = self._get_folder_attempts(folder_path)
        else:
            # List all dates
            for date_folder in os.listdir(self.base_dir):
                folder_path = os.path.join(self.base_dir, date_folder)
                if os.path.isdir(folder_path):
                    attempts.extend(self._get_folder_attempts(folder_path))
                    
        return sorted(attempts, key=lambda x: x['timestamp'])

    def _get_folder_attempts(self, folder_path):
        """Get all unauthorized attempts from a specific folder"""
        attempts = []
        for filename in os.listdir(folder_path):
            if filename.startswith("unauthorized_") and filename.endswith(".jpg"):
                # Parse filename for metadata
                try:
                    # Extract timestamp and confidence from filename
                    # Format: unauthorized_HH_MM_SS_conf_XX.XX.jpg
                    parts = filename.split('_')
                    time = '_'.join(parts[1:4])  # HH_MM_SS
                    confidence = float(parts[5].replace('.jpg', ''))
                    
                    attempts.append({
                        'timestamp': f"{os.path.basename(folder_path)} {time.replace('_', ':')}",
                        'confidence': confidence,
                        'filepath': os.path.join(folder_path, filename)
                    })
                except:
                    continue
                    
        return attempts
