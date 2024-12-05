import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt

# Step 1: Set up the path to the dataset
dataset_dir = 'dataset'  # Directory containing images

# Step 2: Preprocessing and Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to [0, 1]
    rotation_range=40,          # Random rotation
    width_shift_range=0.2,      # Randomly shift images horizontally
    height_shift_range=0.2,     # Randomly shift images vertically
    shear_range=0.2,            # Shear images randomly
    zoom_range=0.2,             # Zoom randomly
    horizontal_flip=True,       # Random horizontal flip
    fill_mode='nearest',        # Fill in pixels after transformation
    validation_split=0.2        # 20% of data will be used for validation
)

# Step 3: Load and generate training and validation data
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),     # Resize all images to 150x150
    batch_size=32,              # Batch size for training
    class_mode='categorical',   # Use categorical labels for multi-class classification
    subset='training'           # Specify this is the training set
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'         # Specify this is the validation set
)

# Step 4: Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5 output classes
])

# Step 5: Compile the model
model.compile(
    loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class
    optimizer='adam',                 # Adam optimizer
    metrics=['accuracy']
)

# Step 6: Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # Number of epochs to train
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Step 7: Evaluate the model
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation accuracy: {val_acc*100:.2f}%")

# Step 8: Save the model for future use
model.save('image_classifier_model.h5')

# Step 9: Make a prediction for a new image after training
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

# Step 10: Example usage: Call the predict function with the path to an image you want to classify
image_path = 'test.jpg'  # Provide the path to the image you want to classify
predicted_class = predict_image(image_path)
