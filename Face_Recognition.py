import os
import cv2
import time
import numpy as np
import pickle

# importing algorithms
from PCA import pca_class

# importing feature extraction classes
from images_to_matrix import images_to_matrix_class
from dataset import dataset_class

# Create model directory if not exists
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# No of images For Training(Left will be used as testing Image)
no_of_images_of_one_person = 9  # Increased training data per person
dataset_obj = dataset_class(no_of_images_of_one_person)

# Data For Training
images_names = dataset_obj.images_name_for_train
y = dataset_obj.y_for_train
no_of_elements = dataset_obj.no_of_elements_for_train
target_names = dataset_obj.target_name_as_array

training_start_time = time.process_time()
img_width, img_height = 50, 50

i_t_m_c = images_to_matrix_class(images_names, img_width, img_height)

scaled_face = i_t_m_c.get_matrix()

print("Processing images...")
print(f"Image dimensions: {img_height}x{img_width}")

# Algorithm
my_algo = pca_class(scaled_face, y, target_names, no_of_elements, 95)  # Increased quality preservation
new_coordinates = my_algo.reduce_dim()

training_time = time.process_time() - training_start_time

# Save model to model directory
model_path = os.path.join(model_dir, "pca_model.pkl")
model_data = {
    'new_bases': my_algo.new_bases,
    'mean_face': my_algo.mean_face,
    'new_coordinates': new_coordinates,
    'y': y,
    'target_names': target_names,
    'no_of_elements': no_of_elements,
    'img_width': img_width,
    'img_height': img_height,
    'quality_percent': 90
}
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"Model saved to {model_path}")

print(f"Training completed in {training_time:.2f} seconds")

# Testing
correct = 0
wrong = 0
i = 0
net_time_of_reco = 0

y_for_test = dataset_obj.y_for_test

for img_path in dataset_obj.images_name_for_test:
    time_start = time.process_time()
    find_name = my_algo.recognize_face(my_algo.new_cord(img_path, img_height, img_width))
    time_elapsed = (time.process_time() - time_start)
    net_time_of_reco += time_elapsed
    rec_y = y_for_test[i]
    rec_name = target_names[rec_y]
    if find_name is rec_name:
        correct += 1
        print("Correct", " Name:", find_name)
    else:
        wrong +=1
        print("Wrong:", " Real Name:", rec_name, "Rec Y:", rec_y, "Find Name:", find_name)
    i+=1

print("Correct", correct)
print("Wrong", wrong)
print("Total Test Images", i)
print("Percent", correct/i*100)
print("Total Person", len(target_names))
print("Total Train Images", no_of_images_of_one_person * len(target_names))
print("Total Time Taken for reco:", time_elapsed)
print("Time Taken for one reco:", time_elapsed/i)
print("Training Time", training_time)
