import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import matplotlib.pyplot as plt

images = np.load('images_other.npy')
labels = np.load('labels_other.npy', allow_pickle=True)

images, _, labels, _ = train_test_split(images, labels, test_size=0.4, random_state=42, stratify = labels)

plt.imshow(images[0])
plt.show()
label_encoder = LabelEncoder()
labels_numeric = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)

# Split the data into training and validation sets
train_images, val_images, train_labels_numeric, val_labels_numeric = train_test_split(images, labels_numeric, test_size=0.2, stratify=labels_numeric, random_state=42)


train_labels_one_hot = tf.one_hot(train_labels_numeric, depth=num_classes)
val_labels_one_hot = tf.one_hot(val_labels_numeric, depth=num_classes)
# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels_one_hot))
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels_one_hot))

# Shuffle and batch the datasets
batch_size = 32
train_dataset = train_dataset.shuffle(len(train_images)).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=1)

es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Define the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()
# Train the model
epochs = 25
history = model.fit(train_dataset, 
                    validation_data=val_dataset, 
                    epochs=epochs, 
                    callbacks=[cp_callback, es_callback])



train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Get the training and validation accuracy values
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save('other_model.h5')


model2 = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

checkpoint_path = "training_checkpoints/cp-0003.ckpt"

model2.load_weights(checkpoint_path)

model2.save('other_model_3')



# Load the model
model = tf.keras.models.load_model('other_model.h5')

loaded_model = tf.keras.models.load_model('other_model_3')

loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Evaluate the model

test_loss, test_acc = loaded_model.evaluate(val_dataset, verbose=2)
print(test_acc)

# Make predictions

predictions = loaded_model.predict(val_dataset)

predictions2 = loaded_model.predict(val_images)

predicted_label =  label_encoder.inverse_transform(np.argmax(predictions2,axis=1))

original_label = label_encoder.inverse_transform(val_labels_numeric)


# Iterate over the samples and plot the images with labels
# val_images_subset = val_images[:10]
# predicted_labels_subset = predicted_label[:10]
# actual_labels_subset = original_label[:10]


# # Plot the images with predicted and actual labels
# fig, axes = plt.subplots(2, 5, figsize=(12, 6))
# axes = axes.flatten()

# for i, (image, predicted_label, actual_label) in enumerate(zip(val_images_subset, predicted_labels_subset, actual_labels_subset)):
#     axes[i].imshow(image)
#     axes[i].set_title(f"Predicted: {predicted_label}\nActual: {actual_label}")
#     axes[i].axis('off')

# plt.tight_layout()
# plt.show()



def plot_images_with_labels(images, predicted_labels, actual_labels, num_images=10):

    num_plots = int(np.ceil(len(images) / num_images))  # Calculate the number of plots needed

    for plot_index in range(num_plots):
        start_index = plot_index * num_images
        end_index = min((plot_index + 1) * num_images, len(images))
        images_subset = images[start_index:end_index]
        predicted_labels_subset = predicted_labels[start_index:end_index]
        actual_labels_subset = actual_labels[start_index:end_index]

        # Plot the images with predicted and actual labels
        fig, axes = plt.subplots(2, len(images_subset) // 2, figsize=(12, 6))
        axes = axes.flatten()

        for i, (image, predicted_label, actual_label) in enumerate(zip(images_subset, predicted_labels_subset, actual_labels_subset)):
            axes[i].imshow(image)
            axes[i].set_title(f"Predicted: {predicted_label}\nActual: {actual_label}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

# Example usage:
plot_images_with_labels(val_images[:100], predicted_label, original_label, num_images=10)

# Convert predictions to class names
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
print(predicted_labels)

# Convert validation labels to class names
val_labels = label_encoder.inverse_transform(val_labels_numeric)
print(val_labels)

# Create a confusion matrix
confusion_matrix = confusion_matrix(val_labels, predicted_labels)
print(confusion_matrix)


# Create a classification report
from sklearn.metrics import classification_report
report = classification_report(val_labels, predicted_labels)

# Print the classification report
print(report)
