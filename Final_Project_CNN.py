import os
import numpy as np
import pandas as pd
from PIL import Image,ImageSequence
from collections import defaultdict

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import InputLayer, BatchNormalization, Dropout, Flatten, Dense, Activation, MaxPooling2D,\
    Conv2D
from tensorflow.keras import Sequential, Input
import tensorflow as tf
import matplotlib.pyplot as plt
from statistics import mean

from tensorflow.keras import utils
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from statistics import stdev

# _____________________________________  General Helper Methods ______________________________

# Purpose:  Pads the array to the dimensions specified
#
# ... arr -> np.array
# ... desired_height -> int
# ... desired_width -> int
#
# Notes:
# ... Does not mutate array
# ... Desired Height and Desired Width have to be >= arrays current
# dimensions
# ... Calls upon np.pad so has all same restrictions
def pad_array(arr, desired_height, desired_width):
    height, width = arr.shape[0], arr.shape[1]
    pad_up = (desired_height - height) // 2
    pad_down = desired_height - height - pad_up

    pad_left = (desired_width - width) // 2
    pad_right = desired_width - width - pad_left
    return np.pad(arr, ((pad_up, pad_down), (pad_left, pad_right), (0, 0)), mode='constant')


# Converts a GIF image to a list of numpy pixel array
#
# ... fpath -> str
# ... frame_arr -> list
#
# Notes:
# ... dimensions will be ( # of frames, 208, 208 )
# ... will place pixel array into frame_arr
# ... calls upon Image library and pad_array method
def gif_to_pixels(fpath, frame_arr):
    im = Image.open(fpath)

    for frame in ImageSequence.Iterator(im):
        frame_arr.append(pad_array(np.array(frame.convert('RGB')), 208, 208))

    return frame_arr


# General helper method for returning a single value of a dataset
def get_col_value(df, col_name):
    if df.shape[0] == 1:
        return df[col_name].values[0]
    else:
        return None


# Helper method for merging a list of data into a single np.array
def merge_data(data):
    all_data = data[0]

    for i in range(1, 10, 1):
        all_data = np.concatenate((all_data, data[i]), axis=0)

    return all_data


# General helper method
def convert_one_hot_to_scalar(l):
    ret = []
    for val in range(len(l)):
        ret.append(l[val][1] > l[val][0])

    return ret


# ______________________________________ Preprocessing / Data Organization _________________________________


# Reads all image data from ML_Final_Project_Dataset
#
# ... root -> str: the filepath for the directory that contains the rest of the dataset/ other directories
#
# Returns: DataFrame containing Name of file, Filepath of file, Image Data Index which is the index you can find
#          the array within the second return
#
#          np.array of all the image pixel arrays
def read_dataset(root, y):
    data = defaultdict(list)
    image_data_arr = []
    y_names = y['ID'].values

    for disk_dir in os.listdir(root):
        if not disk_dir.startswith('.'):

            for subject_dir in os.listdir(os.path.join(root, disk_dir)):
                if not subject_dir.startswith('.'):

                    path = os.path.join(root, disk_dir, subject_dir, 'FSL_SEG') #FSL_SEG    //    'PROCESSED', 'MPRAGE', 'T88_111'
                    for image_path in os.listdir(path):

                        if image_path.endswith('.gif') and image_path[:13] in y_names:
                            full_image_path = os.path.join(path, image_path)
                            image_data_arr = gif_to_pixels(full_image_path, image_data_arr)
                            y_val = y[y['ID'] == image_path[:13]]['AD'].values[0]

                            data['Name'].append(image_path[:13])
                            data['Filepath'].append(os.path.join(disk_dir, subject_dir))
                            data['Image_Data_Idx'].append(len(image_data_arr) - 1)
                            data['Alzheimers'].append(y_val)

    image_data_arr = np.stack(image_data_arr, axis=0)
    df = pd.DataFrame(data=data)
    print(image_data_arr.shape)
    return df, image_data_arr


# Class for easy organization of DataTracker
class DataTracker:
    def __init__(self, data_numpy, data_df):
        self.pixel_data = data_numpy
        self.image_info = data_df
        self.num_examples = self.pixel_data.shape[0]

        self.training_data = None
        self.train_y_labels = None

        self.test_data = None
        self.test_y_labels = None

    # Sets up all of the fields for the class
    #
    # Breakdown:
    #       - training_data -> list of k_folds np.arrays each representing a 'batch'
    #            -- Each batch consists of (num_examples * train_split) // k_folds examples
    #            -- Each batch has shape (# of ex, 208, 208) where 208 represents number of width and height
    #            -- Shape: (10, # of ex, 208, 208)
    #
    #       - training_y_labels -> list of np.array(int)
    #            -- Each list of int within training_y_labels represents the y_vals of each example within batch np.arr
    #            -- Shape: (10, # of ex)
    #
    #       - test_data -> np.array
    #            -- 1 batch of remaining (1 - train_split) pixel data
    #            -- Shape: (num_examples - # of examples, 208, 208)
    #
    #       - test_y_labels -> list of ints
    #            -- Remaining y labels, each label represents y val of one example in test data
    def set_up_examples(self, train_split=0.8, k_folds=10):
        self.image_info = self.image_info.sample(frac=1)
        train_num = int(train_split * self.num_examples)
        num_per_batch = int(train_num // k_folds)

        self.training_data, self.train_y_labels = self.__get_batches(num_per_batch, k_folds)
        self.test_data, self.test_y_labels = self.__df_to_np_batch(self.num_examples - train_num, 0)
        self.test_data = np.stack(self.test_data)

    # Converts the image_info dataframe into the corresponding numpy array
    # and places them in batches of num_per_batch images
    #
    # k_th_batch -> the current batch we are on
    def __df_to_np_batch(self, num_per_batch, k_th_batch):
        images, image_y_vals = [], []

        for i in range(num_per_batch):
            img = self.image_info[self.image_info.index == (k_th_batch * num_per_batch) + i]
            idx = get_col_value(img, 'Image_Data_Idx')
            y_val = get_col_value(img, 'Alzheimers')

            images.append(self.pixel_data[idx])
            image_y_vals.append(y_val)

        return images, image_y_vals

    # Gets k_folds batches totalling approximately train_num examples (not exactly)
    def __get_batches(self, num_per_batch, k_folds):
        batches, y_vals = [], []

        for k in range(k_folds):
            batch, batch_y_vals = self.__df_to_np_batch(num_per_batch, k)

            batch = np.stack(batch, axis=0)
            batch_y_vals = np.stack(batch_y_vals, axis=0)
            batch = batch.reshape((num_per_batch, 208, 208, 3))

            batches.append(batch)
            y_vals.append(batch_y_vals)

        return batches, y_vals


# _______________________________  Data Augmentation __________________________________________________________________


# Increases number of images by applying DataAugmentation filters to the data
def generate_images(x_train, y_train, num_examples, print_graph=True):
    ZOOM = [.97, 0.99]
    BRIGHT_RANGE = [0.8, 1.2]
    HORZ_FLIP = True
    FILL_MODE = "constant"
    DATA_FORMAT = "channels_last"

    dgen = ImageDataGenerator(rescale=1. / 255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, data_format=DATA_FORMAT,
                       fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP, validation_split=0.1)

    train_iter = dgen.flow(x_train, y_train, batch_size=num_examples, subset='training', shuffle=True)
    valid_iter = dgen.flow(x_train, y_train, batch_size=num_examples // 10, subset='validation', shuffle=True)

    train_data, train_labels = train_iter.next()
    valid_data, valid_labels = valid_iter.next()

    while train_data.shape[0] < num_examples:
        next_data, next_labels = train_iter.next()
        train_data = np.concatenate((train_data, next_data), axis=0)
        train_labels = np.concatenate((train_labels, next_labels), axis=0)

    if print_graph:
        plt.subplot(5, 5, 1)
        plt.imshow(x_train[0])
        plt.title('Real Image')

        plt.subplot(5, 5, 2)
        plt.imshow(train_data[0])
        plt.title('Generated Image')
        plt.show()

    assert train_data.shape[0] == train_labels.shape[0]
    assert valid_data.shape[0] == valid_labels.shape[0]

    return train_data, train_labels, valid_data, valid_labels


# Evens out the proportion of Alzheimer's and non-Alzheimer's dataset using SMOTE
def even_out_labels(train_data, train_labels):
    sm = SMOTE()

    train_data, train_labels = sm.fit_resample(train_data.reshape(-1, 208 * 208 * 3), train_labels)
    train_data = train_data.reshape(-1, 208, 208, 3)

    train_labels = utils.to_categorical(train_labels, 2)

    assert train_data.shape[0] == train_labels.shape[0]

    return train_data, train_labels


# _____________________________________ Build Model ________________________________________


# Responsible for creation of a 'Convolutional Block'
def conv_block(model, num_conv_layers, f, k_size, s, p, act, batch_norm=True, kernel='glorot_uniform'):
    for _ in range(num_conv_layers):
        model.add(Conv2D(filters=f, kernel_size=k_size, strides=s, padding=p, activation=act,
                         kernel_initializer=kernel))

    if batch_norm:
        model.add(BatchNormalization())
    model.add(MaxPooling2D())


# Responsible for creation of a 'Dense Block'
def dense_block(model, num_d_layers, units, act='relu', batch_norm=True, drop=0.0, kernel='glorot_uniform'):
    for _ in range(num_d_layers):
        model.add(Dense(units, activation=act, kernel_initializer=kernel))

    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(drop))


# Responsible for creation of convolutional portion of model
def build_conv_blocks(f, tf_batch, k_size=3, s=(1, 1), act='relu', num_c_blocks=3,
                      num_c_layers=2, drop=0.0, kernel='glorot_uniform'):
    model = Sequential()
    model.add(Input(shape=(208, 208, 3)))

    c_units = f
    treat_as_list = len(f) == num_c_blocks
    for c in range(num_c_blocks):
        if treat_as_list:
            c_units = f[c]

        conv_block(model, num_c_layers, c_units, k_size, s, p='same', act=act, batch_norm=tf_batch[c], kernel=kernel)
        model.add(Dropout(drop))

    model.add(Flatten())
    return model


# Reponsible for creation of fully connected portion of model
def build_fully_connected(model, num_d_blocks, num_d_layers, units, tf_batch, act='relu', drop=0.0, kernel='glorot_uniform'):
    d_units = units
    treat_as_list = len(units) == num_d_blocks

    for d in range(num_d_blocks):
        if treat_as_list:
            d_units = units[d]

        dense_block(model, num_d_layers, d_units, act, tf_batch[d + num_c_blocks], drop, kernel)

    model.add(Dense(2, activation='softmax'))
    return model


# _______________________________________ Train / Predict Using Model ____________________________________________


# Helper method for graphing the loss and accuracy of a model over the epochs
def graph_loss_acc(X, history):
    fig, axs = plt.subplots(1, 2)

    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']
    train_acc = history.history['acc']
    valid_acc = history.history['val_acc']

    axs[0].plot(X, train_loss, c='r', label='Train Loss')
    axs[0].plot(X, valid_loss, c='b', label='Validation Loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_title('Loss over Epochs of Model')
    axs[0].legend(loc='upper center')

    axs[1].plot(X, train_acc, c='r', label='Train Accuracy')
    axs[1].plot(X, valid_acc, c='b', label='Validation Accuracy')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_title('Accuracy over Epochs of Model')
    axs[1].legend(loc='upper center')

    plt.show()


# Trains the model once given a set of hyperparameters
def train_model(train_data, train_labels, valid_data, valid_labels, epochs, optimizer, kernel='lecun_uniform'):
    mcp_save = ModelCheckpoint('weights.{epoch:02d}-{val_acc:.2f}.hdf5', save_best_only=True, monitor='val_acc', mode='max')

    cnn = build_conv_blocks(f=filters, drop=drop, num_c_blocks=num_c_blocks, num_c_layers=num_c_layers,
                            tf_batch=tf_batch, kernel=kernel)
    cnn = build_fully_connected(cnn, num_d_blocks=num_d_blocks, num_d_layers=num_d_layers, units=units, drop=drop,
                                tf_batch=tf_batch, kernel=kernel)
    cnn.compile(optimizer=optimizer, loss=tf.losses.BinaryCrossentropy(), metrics=metrics)

    print(cnn.summary())
    history = cnn.fit(train_data, train_labels, validation_data=(valid_data, valid_labels), epochs=epochs, callbacks=[mcp_save])

    graph_loss_acc(list(range(epochs)), history)

    return history


# Produces the accuracy and sensitivity of a model given a checkpoint and a test set
def predict(checkpoint_path, testX, testY):
    cnn = build_conv_blocks(tf_batch=tf_batch, f=filters, drop=drop, num_c_blocks=num_c_blocks,
                            num_c_layers=num_c_layers, kernel=kernel_initializer)
    cnn = build_fully_connected(cnn, tf_batch=tf_batch, num_d_blocks=num_d_blocks, num_d_layers=num_d_layers,
                                units=units, drop=drop, kernel=kernel_initializer)
    cnn.compile(optimizer=optimizer, loss=tf.losses.BinaryCrossentropy(), metrics=metrics)

    cnn.load_weights(checkpoint_path)

    test_labels = utils.to_categorical(testY)
    test_data, test_labels, _, _ = generate_images(testX, test_labels, 240, print_graph=False)

    print(cnn.evaluate(test_data, test_labels))
    predictions = cnn.predict(test_data)

    y_pred = convert_one_hot_to_scalar(predictions)
    y = convert_one_hot_to_scalar(test_labels)
    mets = metrics_analytics(y, y_pred)

    return mets


# ______________________________________ Graph Functions __________________________________________-


# Helper method for graphing a 3D plot with a colorbar representing fourth dimension
def graph_4d(x, y, z, dim4, xlabel, ylabel, zlabel, title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(x, y, z, c=dim4, cmap=plt.hot())
    fig.colorbar(img)

    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    plt.show()


# Helper method for graphing a 3D plot
def graph_3d(x, y, z, xlabel, ylabel, zlabel,title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)

    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    plt.show()


# Helper method for graphing a line
def graph_line(x, y, xlabel, ylabel, title, scatterplot=False, colors=None, marker=None):
    plt.figure(figsize=(8, 8))

    if scatterplot:
        plt.scatter(x, y, c=colors)
    else:
        plt.plot(x, y, marker=marker)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# _____________________________________________ Hyperparameter Tuning ___________________________________________-


# Hyperparameter tuning generally using one for loop
def hyp_param_tuning(train_data, train_labels, valid_data, valid_labels, start=4.0, end=48.0, step=4.0, epochs=40,
                     print_graphs=True, hyp_param_name=None):
    loss_arr, acc_arr, std_arr = [], [], []
    mcp_save = ModelCheckpoint('weights.{epoch:02d}-{val_acc:.2f}.hdf5', save_best_only=True, monitor='val_acc',
                               mode='max')
    X = np.arange(start, end, step)
    max_val_loss = (-1, 1000, 1000)
    max_val_acc = (-1, 0, 0)
    lr_vals = []

    for val in X:
        lr = 0.1 / 10**val
        print(lr)
        lr_vals.append('{0:.2f}'.format(lr))
        optimizer = Adam(learning_rate=lr)

        cnn = build_conv_blocks(tf_batch=tf_batch, f=filters, drop=drop, num_c_blocks=num_c_blocks,
                                num_c_layers=num_c_layers, kernel=kernel_initializer)
        cnn = build_fully_connected(cnn, tf_batch=tf_batch, num_d_blocks=num_d_blocks, num_d_layers=num_d_layers,
                                    units=units, drop=drop, kernel=kernel_initializer)
        cnn.compile(optimizer=optimizer, loss=tf.losses.BinaryCrossentropy(), metrics=metrics)

        print(cnn.summary())
        history = cnn.fit(train_data, train_labels, validation_data=(valid_data, valid_labels), epochs=epochs,
                          callbacks=[mcp_save])
        acc = mean(history.history['val_acc'][-5:])
        loss_diff = abs(mean(history.history['loss'][-5:]) - mean(history.history['val_loss'][-5:]))
        std_diff = stdev(history.history['val_loss'][-5:])
        print(std_diff)

        # dict_keys(['loss', 'acc', 'sens', 'val_loss', 'val_acc', 'val_sens'])

        loss_arr.append(loss_diff)
        acc_arr.append(acc)
        std_arr.append(std_diff)

        if loss_diff < max_val_loss[1]:
            max_val_loss = (val, loss_diff, acc)

        if acc > max_val_acc[2]:
            max_val_acc = (val, loss_diff, acc)

    if print_graphs:
        graph_line(lr_vals, loss_arr, hyp_param_name, 'Loss', 'Loss Over ' + hyp_param_name)
        graph_line(lr_vals, acc_arr, hyp_param_name, 'Accuracy', 'Accuracy Over ' + hyp_param_name)
        graph_line(lr_vals, std_arr, hyp_param_name, 'Standard Deviation of Loss', 'Std Dev of Loss Over ' + hyp_param_name)

    return max_val_loss, max_val_acc

# _________________ Hyper tuning optimizers _______________________

# Helper method for adding important values of a model's history to a list
def record_history(train_data, train_labels, valid_data, valid_labels,
                    optimizer, opt_name, epochs, max_vals, histories):

    print("Optimizer :", opt_name)
    history = train_model(train_data, train_labels, valid_data, valid_labels, epochs, optimizer=optimizer)
    max_vals[opt_name] = mean(history.history['val_acc'][-5:])
    histories[opt_name] = history


# Helper method for graphing important values of a model's history
def chart_histories(histories, epochs):
    X = list(range(epochs))
    fig, axs = plt.subplots(3, 2, figsize=(5, 5))

    for idx, opt_name in enumerate(histories.keys()):
        history = histories[opt_name]

        train_loss = history.history['loss']
        valid_loss = history.history['val_loss']
        train_acc = history.history['acc']
        valid_acc = history.history['val_acc']

        axs[idx, 0].plot(X, train_loss, c='r', label='Train Loss')
        axs[idx, 0].plot(X, valid_loss, c='b', label='Validation Loss')
        axs[idx, 0].set_ylabel('Loss')
        axs[idx, 0].set_title('Loss over Epochs of ' + opt_name)
        axs[idx, 0].legend(loc='upper center')

        axs[idx, 1].plot(X, train_acc, c='r', label='Train Accuracy')
        axs[idx, 1].plot(X, valid_acc, c='b', label='Validation Accuracy')
        axs[idx, 1].set_ylabel('Accuracy')
        axs[idx, 1].set_title('Accuracy over Epochs of ' + opt_name)
        axs[idx, 1].legend(loc='upper center')

    for idx, ax in enumerate(axs.flat):
        if idx == 4 or idx == 5:
            ax.set(xlabel='Epochs')

    plt.show()


# helper method for setting up overall plot of optimizers graph
def chart_comparisons(histories, epochs):
    X = list(range(epochs))
    colors = ['r', 'g', 'b']

    plt.figure(figsize=(7, 7))

    for idx, key in enumerate(histories.keys()):
        acc = histories[key].history['val_acc']
        plt.plot(X, acc, c=colors[idx], label=key)

    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Direct Comparison Between Optimizer Accuracies')

    plt.show()


# Hyperparameter tunes specificially the optimizers
def hyper_tune_optimizer(train_data, train_labels, valid_data, valid_labels, epochs):
    max_vals = defaultdict(float)
    histories = {}

    record_history(train_data, train_labels, valid_data, valid_labels, 'adam', 'Adam', epochs, max_vals, histories)

    record_history(train_data, train_labels, valid_data, valid_labels, RMSprop(), 'RMSprop', epochs, max_vals, histories)

    record_history(train_data, train_labels, valid_data, valid_labels, SGD(), 'SGD', epochs, max_vals, histories)

    print('Charting Histories')
    chart_histories(histories, epochs)
    chart_comparisons(histories, epochs)

    return max_vals


# _________________________________________ Metrics ______________________________________


# Helper method for calculating five metrics
def get_confusion_matrix(y, y_pred):
    unique_classes = set(y) | set(y_pred)
    n_classes = len(unique_classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    pred_pair = list(zip(y, y_pred))

    for i, j in pred_pair:
        matrix[int(i), int(j)] += 1

    return matrix[0, 0], matrix[1, 1], matrix[0, 1], matrix[1, 0]


# Calculates five metrics and stores in a dict: Accuracy, Sensitivity, Specificity, Precision, F1
def metrics_analytics(y, y_pred):
    n = len(y)

    tn, tp, fp, fn = get_confusion_matrix(y, y_pred)

    accuracy = (tp + tn) / n
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = (2 * precision * sensitivity) / (precision + sensitivity)
    met = {'Accuracy': accuracy, 'Sensitivity': sensitivity, 'Specificity': specificity, 'Precision': precision, 'F1': f1}
    return met

# ________________________________________ Main  _______________________________________________________________


# Main:

# Read dataset
dataset_head_path = '/Users/mihir/Desktop/ML_Final_Project_Dataset'
diagnosis = pd.read_csv('./ad_diagnosis (1).csv')
image_info_df, pixel_data_array = read_dataset(dataset_head_path, diagnosis)
image_info_df = image_info_df.sort_values('Name')

# Organize dataset
dt = DataTracker(pixel_data_array, image_info_df)
dt.set_up_examples(train_split=0.8, k_folds=10)

train_data = merge_data(dt.training_data)
train_labels = merge_data(dt.train_y_labels)

# Data Augmentation
train_data, train_labels = even_out_labels(train_data, train_labels)
train_data, train_labels, valid_data, valid_labels = generate_images(train_data, train_labels, 6000, print_graph=False)

# Global Variables
units = [48, 48, 48]
drop = 0.5
filters = [16, 16]
loss = tf.losses.BinaryCrossentropy()
tf_batch = np.zeros(5)
tf_batch[2] = 1 # [0, 0, 1, 0, 0]
num_c_blocks = 2
num_c_layers = 3
num_d_blocks = 3
num_d_layers = 1
lr = 0.0001
optimizer = Adam(learning_rate=lr)
kernel_initializer = 'lecun_uniform'
metrics = [tf.keras.metrics.BinaryAccuracy(name='acc'), tf.keras.metrics.SensitivityAtSpecificity(0.8), tf.keras.metrics.AUC()]

# Hyerparameter Tuning
#print(train_model(train_data[:1201], train_labels[:1201], valid_data[:121], valid_labels[:121], epochs=60, optimizer=optimizer))
#print(hyp_param_tuning(train_data[:1201], train_labels[:1201],
#                       valid_data[:121], valid_labels[:121],
#                       start=0, end=6, step=1, epochs=20, print_graphs=True, hyp_param_name='Learning Rate'))


# Test / Predict values
mets = predict('weights.25-0.92.hdf5', dt.test_data, dt.test_y_labels)
print('Sensitivity: ', mets['Sensitivity'])
print('Accuracy: ', mets['Accuracy'])




