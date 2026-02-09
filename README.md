# 🩺 Explainable AI (XAI) with Grad-CAM

> *"Why did the AI say this is a dog?"*

This project implements **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the decision-making process of a Deep Learning model. It takes a standard "Black Box" model (VGG16) and generates a heatmap showing exactly which pixels influenced the prediction.

## 🖼️ Project Demo
![AI Heatmap](gradcam_result.jpg)
*(Left: Original Image | Right: Heatmap showing the model's focus area)*

## 🎯 Why This Matters
In high-stakes fields like **Healthcare** (diagnosing tumors) or **Finance** (approving loans), high accuracy is not enough. We need **interpretability**.
* If the AI detects a tumor, a doctor needs to know *where* it sees the anomaly.
* This project proves that the model is focusing on relevant features (the dog's face) rather than background noise (the grass).

## 🛠️ Tech Stack
* **Model:** VGG16 (Pre-trained on ImageNet)
* **Technique:** Grad-CAM (Class Activation Mapping)
* **Libraries:** TensorFlow/Keras, OpenCV, Matplotlib
* **Visualization:** Jet Colormap Overlay

## 🚀 How It Works
1.  **Forward Pass:** The image is passed through the CNN.
2.  **Gradient Calculation:** We compute the gradients of the top predicted class with respect to the last convolutional layer.
3.  **Pooling:** We average these gradients to find the "importance" of each feature map.
4.  **Overlay:** We multiply the feature maps by their importance weights to generate the heatmap.

## 💻 Sample Code
```python
# Generate heatmap for a specific layer
heatmap, original_img = get_gradcam_heatmap(model, img_path, layer_name='block5_conv3')

# Overlay on original image
superimposed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
