import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def create_shape(shape_type, size=64, thickness=2):
    # Create a blank white image
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    if shape_type == 'circle':
        center = (size // 2, size // 2)
        radius = size // 3
        cv2.circle(img, center, radius, (0, 0, 0), thickness)
    
    elif shape_type == 'square':
        margin = size // 4
        cv2.rectangle(img, (margin, margin), (size - margin, size - margin), (0, 0, 0), thickness)
    
    elif shape_type == 'rectangle':
        margin_x = size // 5
        margin_y = size // 3
        cv2.rectangle(img, (margin_x, margin_y), (size - margin_x, size - margin_y), (0, 0, 0), thickness)
    
    elif shape_type == 'triangle':
        points = np.array([
            [size // 2, size // 4],  # top
            [size // 4, 3 * size // 4],  # bottom left
            [3 * size // 4, 3 * size // 4]  # bottom right
        ], dtype=np.int32)
        cv2.polylines(img, [points], True, (0, 0, 0), thickness)
    
    return img

def augment_image(img):
    # Random rotation (limited to smaller angles)
    angle = np.random.randint(-30, 30)
    matrix = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1)
    img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    
    # Random scaling (more conservative)
    scale = np.random.uniform(0.9, 1.1)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    img = cv2.resize(img, (64, 64))
    
    # Add minimal noise
    noise = np.random.normal(0, 2, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def extract_features(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features
    win_size = (64, 64)
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    features = hog.compute(gray)
    
    # Add some shape-specific features
    contours, _ = cv2.findContours(cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        features = np.append(features, [area, perimeter])
    
    return features.flatten()

def create_dataset(num_samples=1000):
    shapes = ['circle', 'square', 'rectangle', 'triangle']
    X = []
    y = []
    
    samples_per_shape = num_samples // len(shapes)
    
    for shape_type in shapes:
        for _ in range(samples_per_shape):
            img = create_shape(shape_type)
            for _ in range(2):  # Create 2 augmented versions
                aug_img = augment_image(img.copy())
                features = extract_features(aug_img)
                X.append(features)
                y.append(shapes.index(shape_type))
    
    return np.array(X), np.array(y)

# Create synthetic dataset
print("Creating synthetic dataset...")
X_train, y_train = create_dataset(2000)  # Reduced dataset size
X_test, y_test = create_dataset(500)

# Create and train the model
print("Training model...")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True))
])

model.fit(X_train, y_train)

# Evaluate the model
print("\nEvaluating model...")
test_acc = model.score(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model
print("\nSaving model...")
joblib.dump(model, 'model.joblib')

# Print model size
model_size = os.path.getsize('model.joblib') / (1024 * 1024)  # Size in MB
print(f"\nModel size: {model_size:.2f} MB") 