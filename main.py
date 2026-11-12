import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Set paths: point DATA_DIR at the unzipped Intel Image Classification dataset
# (the folder containing seg_train/, seg_test/ and seg_pred/). Defaults to ./data
base_dir = os.environ.get('DATA_DIR', 'data')
train_dir = os.path.join(base_dir, 'seg_train', 'seg_train')
test_dir = os.path.join(base_dir, 'seg_test', 'seg_test')
pred_dir = os.path.join(base_dir, 'seg_pred', 'seg_pred')


# Image size
img_size = (150, 150)


# Class names (explicitly defined based on your description)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
num_classes = len(class_names)


# Number of images to display per class
num_images_per_class = 5


# Set up the matplotlib figure and axes
fig, axes = plt.subplots(num_classes, num_images_per_class, figsize=(15, num_classes * 2))


for i, class_name in enumerate(class_names):
    class_dir = os.path.join(train_dir, class_name)
    image_files = os.listdir(class_dir)[:num_images_per_class]  # Get first few images
   
    for j, image_file in enumerate(image_files):
        img_path = os.path.join(class_dir, image_file)
        img = load_img(img_path, target_size=(150, 150))
        img_array = img_to_array(img)
        img_array /= 255.0  # Normalize the image
       
        axes[i, j].imshow(img_array)
        axes[i, j].axis('off')
        if j == 0:
            axes[i, j].set_title(class_name, size=16)


plt.tight_layout()
plt.show()


# Count the number of images in each class
class_counts = {class_name: len(os.listdir(os.path.join(train_dir, class_name))) for class_name in class_names}


# Plot the distribution
plt.figure(figsize=(10, 5))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Distribution of Classes in Training Set')
plt.xticks(rotation=45)
plt.show()


# Function to get color channel means
def get_color_means(directory):
    means = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('jpeg', 'jpg', 'png')):
                img_path = os.path.join(subdir, file)
                img = load_img(img_path, target_size=(150, 150))
                img_array = img_to_array(img) / 255.0
                means.append(img_array.mean(axis=(0, 1)))
    return np.array(means)


# Get color channel means for training dataset
color_means = get_color_means(train_dir)


# Convert to DataFrame for easy plotting
color_df = pd.DataFrame(color_means, columns=['Red', 'Green', 'Blue'])


# Plot color channel distributions
plt.figure(figsize=(12, 6))
color_df.plot(kind='box', title='Distribution of Color Channels')
plt.show()


# Function to compute average image for each class
def compute_average_image(directory, target_size=(150, 150)):
    avg_images = {}
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        image_sum = np.zeros((target_size[0], target_size[1], 3))
        count = 0
        for file in os.listdir(class_dir):
            if file.endswith(('jpeg', 'jpg', 'png')):
                img_path = os.path.join(class_dir, file)
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img) / 255.0
                image_sum += img_array
                count += 1
        avg_images[class_name] = image_sum / count
    return avg_images


# Compute average images for each class
avg_images = compute_average_image(train_dir)


# Plot average images
fig, axes = plt.subplots(1, num_classes, figsize=(18, 3))
for i, (class_name, avg_img) in enumerate(avg_images.items()):
    axes[i].imshow(avg_img)
    axes[i].axis('off')
    axes[i].set_title(class_name, size=16)


plt.tight_layout()
plt.show()


# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)


# No augmentation for validation (same 20% split as training) or test
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)


# Create data generators
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=img_size,
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    subset='training')


validation_generator = val_datagen.flow_from_directory(train_dir,
                                                         target_size=img_size,
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         subset='validation')


test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=img_size,
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle=False)


# Define input shape
input_shape = (150, 150, 3)
inputs = Input(shape=input_shape)


# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)


# Freeze the base model initially
for layer in base_model.layers:
    layer.trainable = False


# Unfreeze block 5 (last 3 conv layers + pooling) for fine-tuning
for layer in base_model.layers[-4:]:
    layer.trainable = True


# Add custom layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(6, activation='softmax')(x)


# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)


# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])


# Display the model summary
model.summary()


# Define the callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
model_checkpoint = ModelCheckpoint(os.path.join(base_dir, 'best_model.keras'), save_best_only=True, monitor='val_loss', verbose=1)


# Train the model with the defined callbacks
history = model.fit(train_generator,
                    epochs=50,
                    validation_data=validation_generator,
                    callbacks=[early_stopping, reduce_lr, model_checkpoint])


# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Time')


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Time')


plt.show()


# Step 8: Evaluate the Model
print("Evaluating the model...")


# Evaluate on the training data
train_loss, train_acc = model.evaluate(train_generator, verbose=0)


# Evaluate on the held-out test data
test_loss, test_acc = model.evaluate(test_generator, verbose=0)


# Print the evaluation results in a concise format
print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")


# Predict the test set and print additional details
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())


# Classification report
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(labels, predicted_classes, target_names=class_labels))


# Get the true labels
true_labels = test_generator.classes
# Compute the confusion matrix
cm = confusion_matrix(true_labels, predicted_classes)


# Plot the confusion matrix
plt.figure(figsize=(10, 8))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# Binarize the labels for multi-class ROC
labels_binarized = label_binarize(labels, classes=range(len(class_labels)))


# Compute ROC curve and ROC area for each class
fpr = {}
tpr = {}
roc_auc = {}
n_classes = len(class_labels)


for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(labels_binarized[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(labels_binarized.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Plot ROC curves
plt.figure(figsize=(10, 8))


colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {class_labels[i]}')


plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
         label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})')


plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Multi-class')
plt.legend(loc="lower right")
plt.show() 
