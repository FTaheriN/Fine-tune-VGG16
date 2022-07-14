import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers, models, Model, optimizers
import glob
fail_ = glob.glob('/train/fail/*.*')
pass_ = glob.glob('/train/pass/*.*')

data = []
labels = []

for i in fail_:   
  image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
  target_size= (224,224))
  image=np.array(image)
  data.append(image)
  labels.append('fail')
for i in pass_:   
  image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
  target_size= (224,224))
  image=np.array(image)
  data.append(image)
  labels.append('pass')

train_data = np.array(data)
train_labels = np.array(labels)

# normalize data
X_train = train_data.astype('float32')
X_train /= 255

# one-hot labels
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(train_labels))

# test data
fail2_ = glob.glob('/test/fail/*.*')
pass2_ = glob.glob('/test/pass/*.*')

data2 = []
labels2 = []

for i in fail2_:   
  image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
  target_size= (224,224))
  image=np.array(image)
  data2.append(image)
  labels2.append('fail')
for i in pass2_:   
  image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
  target_size= (224,224))
  image=np.array(image)
  data2.append(image)
  labels2.append('pass')

test_data = np.array(data2)
test_labels = np.array(labels2)

# normalize data
X_test = test_data.astype('float32')
X_test /= 255

# one-hot labels
lb2 = LabelEncoder()
y_test = np_utils.to_categorical(lb2.fit_transform(test_labels))

# importing model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze four convolution blocks
for layer in vgg_model.layers[:15]:
    layer.trainable = False
# Make sure you have frozen the correct layers
for i, layer in enumerate(vgg_model.layers):
    print(i, layer.name, layer.trainable)

x = vgg_model.output
x = Flatten()(x) # Flatten dimensions to for use in FC layers
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
x = Dense(256, activation='relu')(x)
x = Dense(2, activation='softmax')(x) # Softmax for multiclass
transfer_model = Model(inputs=vgg_model.input, outputs=x)

lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)
checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 1)
early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1)

# compile model
learning_rate= 5e-5

transfer_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

history = transfer_model.fit(X_train, y_train, batch_size = 1, epochs=10, validation_data=(X_test,y_test), callbacks=[lr_reduce,checkpoint,early_stopping])

