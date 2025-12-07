import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk

# --- Image Settings ---
image_size = (128, 128)
model_path = r'C:\Users\aasth\OneDrive\Desktop\flower_classification_project\flower_model.h5'
model = load_model(model_path)

# --- Class Labels (update based on your actual class names if needed) ---
class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# --- Prediction Function ---
def predict_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if not file_path:
        return

    # Display selected image
    img = Image.open(file_path).resize((200, 200))
    tk_img = ImageTk.PhotoImage(img)
    image_label.config(image=tk_img)
    image_label.image = tk_img

    # Prepare image for model
    img = load_img(file_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    result_text = f"Prediction: {class_labels[predicted_class]}\nConfidence: {confidence*100:.2f}%"
    result_label.config(text=result_text)

# --- Tkinter GUI Setup ---
window = tk.Tk()
window.title("🌼 Flower Image Classifier")
window.geometry("400x450")
window.configure(bg="#f9f9f9")

title_label = Label(window, text="Flower Image Classifier", font=("Helvetica", 16, 'bold'), bg="#f9f9f9")
title_label.pack(pady=10)

image_label = Label(window, bg="#f9f9f9")
image_label.pack()

browse_button = tk.Button(window, text="Upload Image", command=predict_image, font=("Helvetica", 12), bg="#e0e0e0")
browse_button.pack(pady=10)

result_label = Label(window, text="", font=("Helvetica", 14), bg="#f9f9f9", fg="#333")
result_label.pack(pady=10)

window.mainloop()
