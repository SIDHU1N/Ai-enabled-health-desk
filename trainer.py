import os
import cv2
import numpy as np
import re

# Initialize LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
dataset_path = 'Data'  # Folder containing training images

def get_images_and_labels(path):
    faces = []
    users = []
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))]

    for img_path in image_paths:
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert image to grayscale
        
        # Extract user ID from filename (supports "user_123.jpg", "123.jpg", etc.)
        match = re.search(r"(\d+)", os.path.basename(img_path))
        if not match:
            print(f"Skipping {img_path}: Invalid filename format.")
            continue
        user_id = int(match.group(1))

        faces.append(gray_img)
        users.append(user_id)

        print(f"Processing User ID: {user_id}")
        cv2.imshow("Training", gray_img)
        cv2.waitKey(100)

    return users, faces

# Load images and labels
users, faces = get_images_and_labels(dataset_path)

if faces:
    recognizer.train(faces, np.array(users))  # Train recognizer
    recognizer.save('recognizer/TrainingData.yml')  # Save trained model
    print("✅ Training complete! Model saved as 'TrainingData.yml'.")
else:
    print("⚠️ No valid training data found! Please check your dataset.")

cv2.destroyAllWindows()
