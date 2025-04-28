import numpy as np
import cv2
import scipy.linalg as s_linalg
import pickle
import os


class pca_class:
    def give_p(self, d):
        sum = np.sum(d)
        sum_85 = self.quality_percent * sum/100
        temp = 0
        p = 0
        while temp < sum_85:
            temp += d[p]
            p += 1
        return p

    def reduce_dim(self):
        p, d, q = s_linalg.svd(self.images, full_matrices=True)
        p_matrix = np.matrix(p)

        p = self.give_p(d)
        self.new_bases = p_matrix[:, 0:p]
        self.new_coordinates = np.dot(self.new_bases.T, self.images)
        return self.new_coordinates.T

    def __init__(self, images=None, y=None, target_names=None, no_of_elements=None, quality_percent=90):
        if images is not None:
            self.no_of_elements = no_of_elements
            self.images = np.asarray(images)
            self.y = y
            self.target_names = target_names
            mean = np.mean(self.images, 1)
            self.mean_face = np.asmatrix(mean).T
            self.images = self.images - self.mean_face
            self.quality_percent = quality_percent
            # Initialize registered faces dictionary
            self.registered_faces = {}

    def save_model(self, filepath):
        """Save the trained PCA model to a file"""
        model_data = {
            'new_bases': self.new_bases,
            'mean_face': self.mean_face,
            'registered_faces': self.registered_faces,
            'quality_percent': self.quality_percent
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Load a trained PCA model from a file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls()  # Create empty model
        model.new_bases = model_data['new_bases']
        model.mean_face = model_data['mean_face']
        model.registered_faces = model_data.get('registered_faces', {})
        model.quality_percent = model_data.get('quality_percent', 90)
        return model

    def register_face(self, face_id, face_image, img_height=50, img_width=50):
        """Register a new face in the system"""
        if isinstance(face_image, str):  # If path is provided
            img = cv2.imread(face_image)
            gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (img_height, img_width))
        else:  # If image array is provided
            gray = cv2.resize(face_image, (img_height, img_width))
        
        img_vec = np.asmatrix(gray).ravel().T
        new_coordinates = self.get_face_encoding(img_vec)
        
        self.registered_faces[face_id] = {
            'encoding': new_coordinates,
            'timestamp': np.datetime64('now')
        }
        return True

    def get_face_encoding(self, face_vector):
        """Get PCA encoding for a face vector"""
        centered_face = face_vector - self.mean_face
        return np.dot(self.new_bases.T, centered_face)

    def match_face(self, face_image, img_height=50, img_width=50, min_confidence=5):
        """Match a face against registered faces"""
        if isinstance(face_image, str):  # If path is provided
            img = cv2.imread(face_image)
            gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (img_height, img_width))
        else:  # If image array is provided
            gray = cv2.resize(face_image, (img_height, img_width))
        
        img_vec = np.asmatrix(gray).ravel().T
        query_encoding = self.get_face_encoding(img_vec)
        
        best_match = None
        best_confidence = 0
        threshold = 1200

        for face_id, face_data in self.registered_faces.items():
            stored_encoding = face_data['encoding']
            distance = np.linalg.norm(query_encoding - stored_encoding)
            confidence = max(0, 100 * (1 - distance/threshold))
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = face_id

        if best_confidence >= min_confidence:
            return {
                'match_found': True,
                'face_id': best_match,
                'confidence': best_confidence,
                'registered_at': self.registered_faces[best_match]['timestamp']
            }
        else:
            return {
                'match_found': False,
                'confidence': best_confidence
            }

    def original_data(self, new_coordinates):
        return self.mean_face + (np.dot(self.new_bases, new_coordinates.T))

    def show_eigen_face(self, height, width, min_pix_int, max_pix_int, eig_no):
        ev = self.new_bases[:, eig_no:eig_no + 1]
        min_orig = np.min(ev)
        max_orig = np.max(ev)
        ev = min_pix_int + (((max_pix_int - min_pix_int)/(max_orig - min_orig)) * ev)
        ev_re = np.reshape(ev, (height, width))
        print(f"Eigenface {eig_no} computed - Shape: {ev_re.shape}")

    def new_cord(self, name, img_height, img_width):
        img = cv2.imread(name)
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (img_height, img_width))
        img_vec = np.asmatrix(gray).ravel()
        img_vec = img_vec.T
        new_mean = ((self.mean_face * len(self.y)) + img_vec)/(len(self.y) + 1)
        img_vec = img_vec - new_mean
        return np.dot(self.new_bases.T, img_vec)

    def new_cord_for_image(self, image):
        img_vec = np.asmatrix(image).ravel()
        img_vec = img_vec.T
        new_mean = ((self.mean_face * len(self.y)) + img_vec) / (len(self.y) + 1)
        img_vec = img_vec - new_mean
        return np.dot(self.new_bases.T, img_vec)

    def recognize_face(self, new_cord_pca, k=0):
        classes = len(self.no_of_elements)
        start = 0
        distances = []
        for i in range(classes):
            temp_imgs = self.new_coordinates[:, int(start): int(start + self.no_of_elements[i])]
            mean_temp = np.mean(temp_imgs, 1)
            start = start + self.no_of_elements[i]
            dist = np.linalg.norm(new_cord_pca - mean_temp)
            distances += [dist]
        min = np.argmin(distances)

        # Optimized thresholds for maximum accuracy
        threshold = 1500       # Increased threshold for better feature matching
        min_confidence = 4     # Lower base threshold to reduce rejections
        low_confidence = 8     # Lower threshold for better recall
        med_confidence = 15    # Adjusted medium threshold
        high_confidence = 25   # Adjusted high threshold
        
        # Calculate confidence scores
        distances_copy = np.array(distances.copy())
        best_distance = distances_copy[min]
        confidence_score = max(0, 100 * (1 - best_distance/threshold))
        
        # Find second best match
        distances_copy[min] = float('inf')
        second_min = np.argmin(distances_copy)
        second_distance = distances_copy[second_min]
        second_confidence = max(0, 100 * (1 - second_distance/threshold))
        
        # Dynamic verification thresholds
        min_gap = 1.8 if confidence_score >= med_confidence else 1.3
        confidence_gap = confidence_score - second_confidence
        confidence_ratio = confidence_score / max(second_confidence, 1.0)  # Prevent extreme ratios
        
        # Advanced multi-tier verification
        high_confidence_verify = (
            confidence_score >= high_confidence and (
                (confidence_gap >= 1.0 and confidence_ratio >= 1.2) or  # Relaxed for high confidence
                confidence_gap >= 1.5
            )
        )
        medium_confidence_verify = (
            confidence_score >= med_confidence and (
                (confidence_gap >= 0.8 and confidence_ratio >= 1.15) or  # More permissive
                confidence_gap >= 1.2
            )
        )
        low_confidence_verify = (
            confidence_score >= min_confidence and (
                (confidence_gap >= 0.6 and confidence_ratio >= 1.1 and second_confidence < med_confidence) or
                (confidence_gap >= 1.0 and confidence_ratio >= 1.2)
            )
        )
        
        # Combined verification with strict security
        is_verified = high_confidence_verify or medium_confidence_verify or low_confidence_verify
        
        if best_distance < threshold and confidence_score >= min_confidence and is_verified:
            if confidence_score >= high_confidence:
                confidence_level = "HIGH"
                security_color = "GREEN"
            elif confidence_score >= med_confidence:
                confidence_level = "MEDIUM"
                security_color = "YELLOW"
            elif confidence_score >= low_confidence:
                confidence_level = "LOW"
                security_color = "BLUE"
            else:
                confidence_level = "VERY LOW"
                security_color = "GRAY"
            
            print(f"[{security_color}] Match - Person {k}: {self.target_names[min]}")
            print(f"    Confidence: {confidence_score:.1f}% ({confidence_level})")
            print(f"    Confidence Gap: {confidence_gap:.1f}% (Ratio: {confidence_ratio:.2f})")
            print(f"    ID: {min}")
            return self.target_names[min]
        else:
            reject_reason = "Below minimum confidence" if confidence_score < min_confidence else "Failed verification"
            print(f"[RED] REJECTED - Unknown Person")
            print(f"    Confidence: {confidence_score:.1f}% ({reject_reason})")
            print(f"    Confidence Gap: {confidence_gap:.1f}% (Ratio: {confidence_ratio:.2f})")
            return 'Unknown'
