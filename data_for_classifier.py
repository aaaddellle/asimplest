import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ['LOKY_MAX_CPU_COUNT'] = '8'


import matplotlib.pyplot as plt
import numpy as np
from classifier import model, feature_extraction_model
#from classifier import sequential
from PIL import Image
from keras.models import Model

from sklearn.manifold import TSNE  # Import TSNE
from sklearn.preprocessing import OneHotEncoder

print("libraries downloaded")
data_path = "C:\\Users\\Lenovo\\Desktop\\python\\asimplest\\mels_1"
img_size = (64, 64)

print("data loading start")
def load_data(data_path):
    images = []
    labels = []
    '''for digit in range(10):
        digit_path = os.path.join(data_path, str(digit))
        for filename in os.listdir(digit_path):
            img = Image.open(os.path.join(digit_path, filename)).convert('L')  # Convert to grayscale
            img = img.resize(img_size)
            img = np.array(img)
            img = img.astype('float32') / 255.0  # Normalization
            images.append(img)
            labels.append(digit)
    return np.array(images), np.array(labels)'''

    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Get the label from the folder name
            #label = int(folder_name)
            
            # Loop through each image file in the current folder
            for filename in os.listdir(folder_path):
                img = Image.open(os.path.join(folder_path, filename)).convert('L')  # Convert to grayscale
                img = img.resize(img_size)
                img = np.array(img)
                img = img.astype('float32') / 255.0  # Normalization
                images.append(img)
                labels.append(folder_name)
    
    return np.array(images), np.array(labels)

# Load the data
x_data, y_data = load_data(data_path)
print("data loading done")
# Shuffle the data
indices = np.arange(len(x_data))
np.random.shuffle(indices)
x_data = x_data[indices] 
y_data = y_data[indices]
print("splitting the data")
# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(x_data))
x_train, x_test = x_data[:split_index], x_data[split_index:]
y_train, y_test = y_data[:split_index], y_data[split_index:]
print(f'Shape of the train data: {x_train.shape}')
print(f'Shape of the test data: {x_test.shape}')

encoder = OneHotEncoder(sparse=False)
y_train_encoded  = encoder.fit_transform(y_train.reshape(-1,1))
y_test_encoded = encoder.fit_transform(y_test.reshape(-1, 1))
y_test_labels = y_test_encoded.argmax(axis=1)

import random
random_indices = random.sample(range(len(x_train)), 10)
for i in random_indices:
    plt.imshow(x_train[i], cmap='gray')
    plt.xlabel(f"Label: {y_train[i]}")
    #plt.show()


print("model start")
history = model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), epochs=30, batch_size=40)


print("evaluation start")
test_loss, test_acc = model.evaluate(x_test, y_test_encoded)
print("Test accuracy:", test_acc)

print("predictions start")
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

print("feature extraction start")
extracted_features = feature_extraction_model.predict(x_test)

print("======> Extracted Features:")
print(extracted_features)
print(extracted_features.shape)
import pandas as pd

reshaped_features = extracted_features.reshape(extracted_features.shape[0], -1)
features_df = pd.DataFrame(reshaped_features)

file_path = "C:\\Users\\Lenovo\\Desktop\\python\\asimplest\\features.json"
features_df.to_json(file_path, orient='records')
print("extracted features saved to:", file_path)



print("printing random 10 test images")

# Assuming y_train contains the original string labels
unique_labels = np.unique(y_train)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}


string_labels = [index_to_label[idx] for idx in predicted_classes]
true_class_labels = [index_to_label[idx] for idx in y_test_labels]

# Use string_labels for any post-processing or results display
print("Sample Predictions with Original Labels:")
for i in range(10):
    plt.imshow(x_test[i], cmap='gray')
    plt.xlabel(f"True class: {true_class_labels[i]}, Predicted class: {string_labels[i]}")
    plt.show()


top1_correct = np.sum(predicted_classes == y_test_labels)
top1_accuracy = top1_correct / len(y_test_labels) * 100
top1_error = 100 - top1_accuracy
print("Top-1 accuracy: {:.2f}%".format(top1_accuracy))


top5_correct = 0
for i in range(len(y_test_labels)):
    top5_predictions = np.argsort(predictions[i])[-5:][::-1]
    if y_test_labels[i] in top5_predictions:
        top5_correct += 1

top5_accuracy = top5_correct / len(y_test_labels) * 100
top5_error = 100 - top5_accuracy
print("Top-5 accuracy: {:.2f}%".format(top5_accuracy))
print("Top-5 error: {:.2f}%".format(top5_error))

print(predicted_classes.shape, y_test_encoded.shape)
print(predicted_classes.dtype, y_test_encoded.dtype)


penultimate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
features = penultimate_layer_model.predict(x_data)

tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(features)

plt.figure(figsize=(10, 8))
unique_labels = np.unique(y_data)
for label in unique_labels:
    label_indices = np.where(y_data == label)[0]
    plt.scatter(tsne_features[label_indices, 0], tsne_features[label_indices, 1], label = label)


plt.title('t-SNE Visualization of Digit Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()
from sklearn.metrics import accuracy_score

# Assuming 'predicted_classes' are the predicted labels from your model
# and 'y_test' are the actual labels
# Make sure both are in the same format, e.g., both are string labels or both are indexed labels.

unique_labels = np.unique(y_test)  # All unique bird categories
accuracies = {}

for label in unique_labels:
    # Find indices where the current label is the true label
    indices = np.where(y_test == label)
    # Subset the predicted classes at these indices
    label_preds = predicted_classes[indices]
    # Calculate accuracy for the current label
    label_acc = accuracy_score(y_test_labels[indices], label_preds)
    accuracies[label] = label_acc
    print(f"{label}: Accuracy = {label_acc * 100:.2f}%")
