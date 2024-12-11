import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import matplotlib.pyplot as plt

# Path to the trained model
MODEL_PATH = "Crop Disease Prediction Model_CNN.h5"

# Load the trained model
model = load_model(MODEL_PATH)

# Define the classes (update these based on your trained model's classes)
DISEASE_CLASSES = ['Black Gram_Anthracnose', 'Black Gram_Healthy', 'Black Gram_Leaf Crinckle',
                   'Black Gram_Powdery Mildew', 'Black Gram_Yellow Mosaic', 'Rice_Bacterial Blight',
                   'Rice_BrownSpot', 'Rice_False Smut', 'Rice_Healthy', 'Rice_LeafBlast',
                   'Rice_Sheath Blight', 'Rice_Tungro']

# Function to predict and display all class probabilities from the raw image
def predict_and_analyze(image_path):
    """
    Predicts the class probabilities for the uploaded raw image and displays the distribution.
    Args:
        image_path (str): Path to the input image.
    Returns:
        dict: Class probabilities for each disease.
    """
    # Load the image as-is without preprocessing
    img = load_img(image_path)  # Load image without any preprocessing
    img_array = np.array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class probabilities
    predictions = model.predict(img_array)[0]  # Get the probabilities for the image
    class_probabilities = {DISEASE_CLASSES[i]: predictions[i] for i in range(len(DISEASE_CLASSES))}
    
    # Plot the probability distribution
    plt.figure(figsize=(10, 5))
    plt.bar(class_probabilities.keys(), class_probabilities.values(), color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Probability")
    plt.title("Class Probability Distribution")
    plt.tight_layout()
    plt.show()

    return class_probabilities

# Upload and test the model
if __name__ == "__main__":

    # Path to the image for testing
    test_image_path = '1p.jpg'  # Example: 'path_to_image.jpg'

    try:
        # Get class probabilities and analyze
        class_probabilities = predict_and_analyze(test_image_path)

        # Display the image
        img = load_img(test_image_path)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Input Image")
        plt.show()

        print("Class Probabilities:")
        for disease, prob in class_probabilities.items():
            print(f"{disease}: {prob * 100:.2f}%")

    except Exception as e:
        print(f"Error: {e}")