# Customized def 
import models 

import os
from glob import glob 
import cv2 
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import time

base_path = 'C:/Users/user/Desktop/datasets/isic_images'
base_models_path = 'C:/Users/user/Desktop/models/isic_images'
N_RES = 256

learning_rate = 0.001
batch_size = 32
hidden_units = 512
projection_units = 128
num_epochs = 50
dropout_rate = 0.5
temperature = 0.05
input_shape = (256, 256, 3)
num_classes = 3

labels_idx = {
    'benign' : 0 , 
    'indeterminate' : 1,
    'malignant' : 2,
}


images_dataset = glob(os.path.join(base_path, '*.JPG'))
labels_dataset = pd.read_csv(os.path.join(base_path, 'metadata.csv')).dropna(subset=['benign_malignant'])


images = [] 
labels = [] 
for data in images_dataset:
    img = cv2.imread(data, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (N_RES, N_RES))
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # idx = labels_dataset[labels_dataset[]]
    filename = data.split('\\')[-1][:-4]
    lbl = labels_dataset[labels_dataset['isic_id'] == filename]['benign_malignant']
    lbl = str(lbl.values[0])
    
    # lbl = lbl.to_string()
    
    
    images.append(img) 
    # labels.append(tf.keras.utils.to_categorical(labels_idx[lbl], num_classes))
    labels.append(labels_idx[lbl])
    
# for idx, val in labels_dataset.iterrows():
    # print(val)
    # labels.append(val['melanocytic'])
    # print(val['benign_malignant'])
    # if val['benign_malignant'] is None:
    #     pass
    
    # if val['melanocytic'] is NaN:
    # idx = labels_idx[val['benign_malignant']]
    # lbl = tf.keras.utils.to_categorical(idx, num_classes)
    # labels.append(lbl) 


random.shuffle(images)
random.shuffle(labels)

images = np.reshape(images, [-1, N_RES, N_RES, 3])
labels = np.reshape(labels, [-1, 1])
    
    
test_rate = int(len(images) * 0.3)
x_test = images[:test_rate]
x_train = images[test_rate:]

y_test = labels[:test_rate]
y_train = labels[test_rate:]

# print(x_train.shape)
# print(y_train.shape)
    
# x_train, y_train, x_test, y_test = train_test_split(
#     images, 
#     labels, 
#     random_state=1, 
#     test_size=0.3
# )

data_augmentation = keras.Sequential([
            layers.Normalization(),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.02),
            layers.RandomWidth(0.2),
            layers.RandomHeight(0.2),
        ])

data_augmentation.layers[0].adapt(x_train)

encoder = models.create_encoder(input_shape, data_augmentation)

classifier = models.create_classifier(encoder, input_shape, learning_rate, dropout_rate, hidden_units, num_classes)


# 
encoder = models.create_encoder(input_shape, data_augmentation)

encoder_with_projection_head = models.add_projection_head(encoder, input_shape, projection_units)

encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=models.SupervisedContrastiveLoss(temperature),
)

# encoder_with_projection_head.summary()

history = encoder_with_projection_head.fit(
    x=x_train, 
    y=y_train, 
    batch_size=batch_size, 
    epochs=num_epochs
)

classifier = models.create_classifier(encoder, input_shape, learning_rate, dropout_rate, hidden_units, num_classes, trainable=False)

history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f"Test accuracy: {round(accuracy * 100, 2)}%")


# evaluation
classifier.save(f'{base_models_path}/{time.strftime("%Y%m%d-%H%M%S")}_isic_classification.h5')

# import pandas as pd
hist_df = pd.DataFrame(history.history)
with open(f'{base_models_path}/{time.strftime("%Y%m%d-%H%M%S")}_isic_classification.csv', mode='w') as f:
    hist_df.to_csv(f)

# aug = 


# model = models.create_model()
# print(model)
