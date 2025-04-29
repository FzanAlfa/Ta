import cv2
import numpy as np
import pickle
import os
import shutil
from datetime import datetime
from PCA import pca_class
from screenshot import SecurityLogger

class FaceRecognitionSystem:
    def __init__(self):
        """
        Initialize the Face Recognition System.
        The system maintains two models:
        1. original_model.pkl - Original PCA model without registered faces
        2. registered_model.pkl - Model containing all registered faces
        """
        self.model_dir = 'model'
        self.original_model_path = os.path.join(self.model_dir, 'original_model.pkl')
        self.registered_model_path = os.path.join(self.model_dir, 'registered_model.pkl')
        
        # Create model directory if not exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Initialize security logger
        self.security_logger = SecurityLogger()
        
        # Initialize or load models
        self._initialize_models()
            
        # Load face cascade classifier
        cascade_path = 'cascades/data/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def _initialize_models(self):
        """Initialize both original and registered models"""
        # If no models exist, copy existing pca_model.pkl
        if not os.path.exists(self.original_model_path):
            print("Initializing models from existing PCA model...")
            if os.path.exists('model/pca_model.pkl'):
                shutil.copy2('model/pca_model.pkl', self.original_model_path)
                shutil.copy2('model/pca_model.pkl', self.registered_model_path)
                self.model = pca_class.load_model(self.registered_model_path)
            else:
                print("Error: No existing PCA model found. Please run Face_Recognition.py first to create the initial model.")
                exit(1)
        else:
            # Load registered model if exists, otherwise copy from original
            if os.path.exists(self.registered_model_path):
                print("Loading existing registered faces model...")
                self.model = pca_class.load_model(self.registered_model_path)
            else:
                print("Creating new registered faces model from original...")
                shutil.copy2(self.original_model_path, self.registered_model_path)
                self.model = pca_class.load_model(self.registered_model_path)

    def verify_user(self):
        """Verify a user's face"""
        print("\nStarting user verification...")
        print("Position your face in front of the camera")
        print("Press SPACE to capture your face or Q to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        face = None
        frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Face Verification - Press SPACE to capture, Q to quit', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                face = None
                break
            elif key == ord(' ') and len(faces) > 0:
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face
                face = gray[y:y+h, x:x+w]
                break

        cap.release()
        cv2.destroyAllWindows()

        if face is None:
            print("Verification cancelled or no face detected")
            return

        try:
            result = self.model.match_face(face)
            
            if result['match_found']:
                print("\n=== Match Found! ===")
                print(f"User ID: {result['face_id']}")
                print(f"Confidence: {result['confidence']:.2f}%")
                print(f"Registration Date: {result['registered_at']}")
            else:
                print("\n=== No Match Found ===")
                print(f"Confidence: {result['confidence']:.2f}%")
                # Log unauthorized access attempt
                self.security_logger.save_unauthorized_attempt(frame, result['confidence'])
                
        except Exception as e:
            print(f"Error during verification: {str(e)}")
            # Log error case
            if frame is not None:
                self.security_logger.save_unauthorized_attempt(frame, 0.0)

    def register_user(self):
        """
        Register a new user by:
        1. Capturing their face using webcam
        2. Processing the face image to extract features
        3. Storing the face data in registered_model.pkl
        
        The original_model.pkl remains unchanged as a backup
        """
        print("\n=== User Registration Process ===")
        print("1. Position your face in front of the camera")
        print("2. Press SPACE to capture your face or Q to quit")
        print("3. Enter your ID when prompted")
        print("\nNote: Your face data will be stored in registered_model.pkl")
        
        face = self.capture_face()
        if face is None:
            print("Registration cancelled or no face detected")
            return False

        user_id = input("Enter user ID (name or number): ").strip()
        if not user_id:
            print("Invalid user ID")
            return False

        try:
            success = self.model.register_face(user_id, face)
            if success:
                print(f"\nSuccessfully registered user: {user_id}")
                print("Face data has been stored in registered_model.pkl")
                print(f"Total registered users: {len(self.model.registered_faces)}")
                # Save only to registered model, keeping original intact
                self.model.save_model(self.registered_model_path)
                return True
            else:
                print("Failed to register face")
                return False
        except Exception as e:
            print(f"Error during registration: {str(e)}")
            return False

    def capture_face(self):
        """Capture face from webcam"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Face Registration - Press SPACE to capture, Q to quit', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                face = None
                break
            elif key == ord(' ') and len(faces) > 0:
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face
                face = gray[y:y+h, x:x+w]
                break

        cap.release()
        cv2.destroyAllWindows()
        return face

    def list_users(self):
        """List all registered users"""
        if not self.model.registered_faces:
            print("\nNo users registered yet.")
            return
        
        print("\n=== Registered Users ===")
        print(f"Total users: {len(self.model.registered_faces)}")
        print("\nUser ID\t\tRegistration Date")
        print("-" * 40)
        for user_id, data in self.model.registered_faces.items():
            print(f"{user_id}\t\t{data['timestamp']}")

    def list_unauthorized_attempts(self):
        """List all unauthorized access attempts"""
        print("\n=== Unauthorized Access Attempts ===")
        attempts = self.security_logger.list_unauthorized_attempts()
        
        if not attempts:
            print("No unauthorized attempts recorded.")
            return
            
        print(f"Total attempts: {len(attempts)}")
        print("\nTimestamp\t\t\tConfidence\tFile")
        print("-" * 70)
        
        for attempt in attempts:
            print(f"{attempt['timestamp']}\t{attempt['confidence']:.2f}%\t{attempt['filepath']}")

    def reset_registration(self):
        """Reset to original model, removing all registered faces"""
        confirm = input("\nWARNING: This will remove all registered users. Continue? (y/n): ")
        if confirm.lower() == 'y':
            shutil.copy2(self.original_model_path, self.registered_model_path)
            self.model = pca_class.load_model(self.registered_model_path)
            print("All registered faces have been removed.")
            print("System reset to original state.")
        else:
            print("Reset cancelled.")

def main():
    """Main CLI interface"""
    system = FaceRecognitionSystem()
    
    while True:
        print("\n=== Face Recognition System ===")
        print("1. Register New User")
        print("2. Verify User")
        print("3. List Registered Users")
        print("4. List Unauthorized Attempts")
        print("5. Reset Registration Data")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            system.register_user()
        elif choice == '2':
            system.verify_user()
        elif choice == '3':
            system.list_users()
        elif choice == '4':
            system.list_unauthorized_attempts()
        elif choice == '5':
            system.reset_registration()
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
