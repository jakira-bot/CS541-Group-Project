import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('image_classifier_model.h5')

def predict_image(image_path):
    # Load the image and resize it to the same size as the training data (150x150)
    img = load_img(image_path, target_size=(150, 150))
    
    # Convert the image to a numpy array and normalize it
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    # Predict the class probabilities
    predictions = model.predict(img_array)
    
    # Get the predicted class (index with highest probability)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Get the class labels
    class_labels = list(train_generator.class_indices.keys())  # These are the folder names (class labels)
    
    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]
    
    # Display the image and predicted class label
    plt.imshow(img)
    plt.title(f"Predicted Class: {predicted_class_label}")
    plt.axis('off')
    plt.show()
    
    # Display the predicted probabilities for each class
    print(f"Predicted Class: {predicted_class_label}")
    print("Class probabilities:")
    
    for i, class_name in enumerate(class_labels):
        print(f"{class_name}: {predictions[0][i]*100:.2f}%")
    
    return predicted_class_label

# Example usage: Call the predict function with the path to an image you want to classify
image_path = 'test.jpg'  
predicted_class = predict_image(image_path)