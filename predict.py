#!/usr/bin/env python

import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import applications, models, preprocessing, metrics
from tensorflow.keras.metrics import AUC
import keras.applications.efficientnet

seed = 0
classes = ['NORMAL', 'PNEUMONIA', 'COVID-19']

test_file_count = 0

for _, _, files in os.walk(os.path.join('./dataset/test/', 'images')):
    test_file_count += len(files)

datagen = keras.preprocessing.image.ImageDataGenerator(
    dtype=float,
    data_format='channels_last',
    preprocessing_function=applications.efficientnet.preprocess_input)

test_it = datagen.flow_from_directory(directory=os.path.join('dataset/test/', 'images'),
                                      target_size=(300, 300),
                                      batch_size=16,
                                      classes=['NORMAL', 'PNEUMONIA', 'COVID-19'],
                                      shuffle=True,
                                      color_mode='grayscale',
                                      class_mode='categorical',
                                      seed=seed)

checkpoints = os.listdir('./model/checkpoints')

# Load latest checkpoint
model = models.load_model(os.path.join('./model/checkpoints', sorted(checkpoints, reverse=True)[0]))
print('Loaded model from epoch: {}'.format(sorted(checkpoints, reverse=True)[0].split(':')[1]))

results = model.evaluate(
    test_it,
    batch_size=16,
    verbose=1,
    steps=int(test_file_count / 16),
    callbacks=None,
    max_queue_size=40,
    workers=8,
    use_multiprocessing=True
)
