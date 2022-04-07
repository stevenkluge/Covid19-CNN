#!/usr/bin/env python
import os
import glob
import argparse
import keras.applications.efficientnet
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.metrics import AUC
from keras.callbacks import LearningRateScheduler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import preprocessing, layers, metrics, models, applications
from tensorflow.keras.utils import plot_model


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    # Start Arguments
    arg_parser.add_argument('--epochs',
                            nargs='?',
                            type=int,
                            default=1,
                            metavar='Integer',
                            help='Number of Epochs to train for')
    arg_parser.add_argument('--lr',
                            nargs='?',
                            type=float,
                            default=1e-3,
                            metavar='Float',
                            help='Starting learning rate for this session')
    # End Arguments
    return arg_parser.parse_args()


# USER VARIABLES
seed = 0

train_file_count = 0
val_file_count = 0
test_file_count = 0
train_normal = 0
train_pneumonia = 0
train_covid = 0
loaded_epoch = 0

for _, _, files in os.walk(os.path.join('./dataset/train/', 'sampled', 'NORMAL')):
    train_normal += len(files)
for _, _, files in os.walk(os.path.join('./dataset/train/', 'sampled', 'PNEUMONIA')):
    train_pneumonia += len(files)
for _, _, files in os.walk(os.path.join('./dataset/train/', 'sampled', 'COVID-19')):
    train_covid += len(files)
for _, _, files in os.walk(os.path.join('./dataset/train/', 'sampled')):
    train_file_count += len(files)
for _, _, files in os.walk(os.path.join('./dataset/validate/', 'images')):
    val_file_count += len(files)
for _, _, files in os.walk(os.path.join('./dataset/test/', 'images')):
    test_file_count += len(files)
print('NORMAL: {}, PNEUMONIA: {}, COVID-19: {}'.format(train_normal, train_pneumonia, train_covid))

'''def get_class_weights(classes: dict):
    total_samples = sum(classes.values())

    class_weights = classes

    for item in classes.items():
        class_weights[item[0]] = total_samples / (len(classes) * item[1])

    return class_weights'''

'''def decay_schedule(epoch, lr):
    # decay lr after every 10 epochs
    if (epoch % 10 == 0) and (epoch != 0):
        lr = lr * 0.1
    return lr'''

args = parse_arguments()

# Define ImageDataGenerator
datagen = keras.preprocessing.image.ImageDataGenerator(
    dtype=float,
    data_format='channels_last',
    preprocessing_function=applications.efficientnet.preprocess_input)

# flow images from directory to avoid loading entire dataset to memory at once
# CHANGE datagen_val to datagen to resume data augmentation
train_it = datagen.flow_from_directory(directory=os.path.join('dataset/train/', 'sampled'),
                                       target_size=(300, 300),
                                       batch_size=16,
                                       classes=['NORMAL', 'PNEUMONIA', 'COVID-19'],
                                       shuffle=True,
                                       color_mode='rgb',
                                       class_mode='categorical',
                                       seed=seed)
val_it = datagen.flow_from_directory(directory=os.path.join('dataset/validate/', 'images'),
                                     target_size=(300, 300),
                                     batch_size=16,
                                     classes=['NORMAL', 'PNEUMONIA', 'COVID-19'],
                                     shuffle=True,
                                     color_mode='rgb',
                                     class_mode='categorical',
                                     seed=seed)

metrics = [
    metrics.Recall(class_id=0, name='normal-recall'),
    metrics.Precision(class_id=0, name='normal-precision'),
    metrics.Recall(class_id=1, name='pneumonia-recall'),
    metrics.Precision(class_id=1, name='pneumonia-precision'),
    metrics.Recall(class_id=2, name='covid-recall'),
    metrics.Precision(class_id=2, name='covid-precision'),
    metrics.FalsePositives(name='false-positives'),
    metrics.FalseNegatives(name='false-negatives'),
    metrics.CategoricalAccuracy()
]

callbacks = [
    keras.callbacks.ModelCheckpoint(
        os.path.join('./model/checkpoints', 'epoch:{epoch:04d}'),
        monitor='val_false-positives',
        mode='min'),
    keras.callbacks.TerminateOnNaN(),
    # keras.callbacks.LearningRateScheduler(decay_schedule, verbose=1),
    keras.callbacks.CSVLogger('model/model_history.csv', separator=',', append=True),
    keras.callbacks.EarlyStopping(monitor='val_false-negatives',
                                  patience=10,
                                  mode='min',
                                  verbose=1,
                                  min_delta=20,
                                  restore_best_weights=False),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_false-negatives',
        factor=0.05,
        patience=5,
        verbose=1,
        mode='min',
        cooldown=0,
        min_lr=1e-10
    )
]

checkpoints = os.listdir('./model/checkpoints')

# If no checkpoints exist, create a new model
base_model = None

if len(checkpoints) == 0:
    # Create Model base using EfficientNetB3

    base_model = keras.applications.efficientnet.EfficientNetB3(
        include_top=False,
        input_shape=(300, 300, 3),
        weights='imagenet',
        classes=3,
        pooling='max')

    x = base_model.output
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(3, activation='softmax')(x)
    model = keras.Model(inputs=base_model.input, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

else:
    # Load latest checkpoint
    model = models.load_model(os.path.join('./model/checkpoints', sorted(checkpoints, reverse=True)[0]))
    loaded_epoch = sorted(checkpoints, reverse=True)[0].split(':')[1]
    print('Loaded model from epoch: {}'.format(sorted(checkpoints, reverse=True)[0].split(':')[1]))

# Set initial Learning Rate
K.set_value(model.optimizer.learning_rate, args.lr)

print('Current Learning Rate: {}'.format(K.get_value(model.optimizer.learning_rate)))
model.fit(
    train_it,
    initial_epoch=int(loaded_epoch),
    epochs=int(loaded_epoch) + args.epochs,
    batch_size=16,
    steps_per_epoch=int(train_file_count / 16),
    validation_data=val_it,
    validation_batch_size=16,
    validation_steps=int(val_file_count / 16),
    callbacks=callbacks,
    workers=8,
    verbose=1,
    use_multiprocessing=True
)
