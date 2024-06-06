import tensorflow as tf
from keras import layers, models

# Set paths to your training and test data
train_dir ="C:/Users/rahum/OneDrive - Iowa State University/teachable machine/train"
test_dir = "C:/Users/rahum/OneDrive - Iowa State University/teachable machine/test"

# Load the training data
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(28, 28),  # Resize images to 28x28
    color_mode='grayscale',
    batch_size=32,
    label_mode='int'
)

# Load the test data
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(28, 28),  # Resize images to 28x28
    color_mode='grayscale',
    batch_size=32,
    label_mode='int'
)

# Build the CNN model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')

])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=5, validation_data=test_dataset)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc}")
