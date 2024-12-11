import React, { useState } from "react";
import { View, Text, Button, Image, StyleSheet, Alert } from "react-native";
import * as ImagePicker from "expo-image-picker"; // For file picking
import axios from "axios";

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictionResult, setPredictionResult] = useState("");

  // Your FastAPI backend URL
  const baseURL = "http://localhost:8000/predict"; // Update this with your backend's actual URL

  // Function to pick an image
  const pickImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert("Permission Denied", "Please allow access to your photo library to upload images.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri); // Store the selected image URI
    }
  };

  // Function to send the selected image to the backend for prediction
  const predictDisease = async () => {
    if (!selectedImage) {
      Alert.alert("No Image Selected", "Please select an image to predict.");
      return;
    }

    const formData = new FormData();
    formData.append("file", {
      uri: selectedImage,
      name: "image.jpg",
      type: "image/jpeg",
    });

    try {
      const response = await axios.post(baseURL, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPredictionResult(response.data.predicted_class); // Extract prediction result from backend response
    } catch (error) {
      console.error(error);
      Alert.alert("Prediction Failed", "An error occurred while predicting the disease.");
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Crop Disease Prediction</Text>

      <Button title="Choose Image" onPress={pickImage} />
      {selectedImage && (
        <Image source={{ uri: selectedImage }} style={styles.image} />
      )}

      <Button title="Predict Disease" onPress={predictDisease} />

      {predictionResult ? (
        <Text style={styles.result}>
          Predicted Disease: {predictionResult}
        </Text>
      ) : null}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, justifyContent: "center", alignItems: "center" },
  header: { fontSize: 24, fontWeight: "bold", marginBottom: 20 },
  image: { width: 200, height: 200, marginVertical: 20 },
  result: { fontSize: 18, marginTop: 20, color: "green" },
});

export default App;
