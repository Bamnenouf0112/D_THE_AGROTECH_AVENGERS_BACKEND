import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the trained model
MODEL_PATH = "Crop Disease Prediction Model_CNN.h5"

# Load the trained model
model = load_model(MODEL_PATH)

# Define the classes (update these based on your trained model's classes)
DISEASE_CLASSES = [
    'Black Gram_Anthracnose', 'Black Gram_Healthy', 'Black Gram_Leaf Crinckle',
    'Black Gram_Powdery Mildew', 'Black Gram_Yellow Mosaic',
    'Rice_Bacterial Blight', 'Rice_BrownSpot', 'Rice_False Smut',
    'Rice_Healthy', 'Rice_LeafBlast', 'Rice_Sheath Blight', 'Rice_Tungro'
]  # Replace with actual class names

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses an image for prediction.
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size to resize the image (default: (224, 224)).
    Returns:
        numpy.ndarray: Preprocessed image ready for model input.
    """
    img = load_img(image_path, target_size=target_size)  # Load and resize image
    img_array = img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Function to predict the disease
def predict_disease(image_path):
    """
    Predicts the class of the uploaded image.
    Args:
        image_path (str): Path to the input image.
    Returns:
        tuple: Predicted class and confidence score.
    """
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)  # Index of class with highest probability
    predicted_class = DISEASE_CLASSES[predicted_class_idx]  # Class name
    return predicted_class_idx, predicted_class

# Generate confusion matrix
def generate_confusion_matrix(test_images, true_labels):
    """
    Generates and displays a confusion matrix.
    Args:
        test_images (list): List of paths to test images.
        true_labels (list): List of true labels corresponding to the test images.
    """
    predicted_labels = []

    # Predict for each test image
    for image_path in test_images:
        predicted_class_idx, _ = predict_disease(image_path)
        predicted_labels.append(predicted_class_idx)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Classification report
    report = classification_report(true_labels, predicted_labels, target_names=DISEASE_CLASSES)
    print("\nClassification Report:")
    print(report)

# Test setup
if __name__ == "__main__":

    # Example: List of test image paths
    test_image_paths = [
        '1a.jpg',  # Replace with actual paths to test images
        '1h.jpg',
        '1l.jpg',
        '1p.jpg',
        '1y.jpg',
        '20231006_164208.jpg',
        '20231006_165550.jpg',
        'aug_0_3.jpg',
        '20231006_163255.jpg',
        '20231006_164402.jpg',
        'aug_0_2.jpg',
        'aug_0_11.jpg',

        # Add more test image paths
    ]

    # Example: Corresponding true labels (indices of DISEASE_CLASSES)
    true_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Replace with actual indices for true labels (e.g., Rice_Healthy=8)

    # Generate confusion matrix
    try:
        generate_confusion_matrix(test_image_paths, true_labels)
    except Exception as e:
        print(f"Error: {e}")
