# !git clone "https://github.com/convman/Multimodal-MOSEI.git"
# !cd Multimodal-MOSEI/
# !chmod +x data/dataset_download.sh
# !./data/dataset_download.sh

#change emotion name accordingly and run in notebook.

import pandas as pd
import numpy as np
import h5py

Test_labels_sad = pd.read_csv("mosi2uni_Test_labels_sad.csv",header=None)
Train_labels_sad = pd.read_csv("mosi2uni_Train_labels_sad.csv",header=None)

video_train = h5py.File("video_train.h5","r")
print(list(video_train.keys()))
video_train = np.array(video_train.get('d1'))

video_test = h5py.File("video_test.h5","r")
print(list(video_test.keys()))
video_test = np.array(video_test.get('d1'))

audio_train = h5py.File("audio_train.h5","r")
print(list(audio_train.keys()))
audio_train = np.array(audio_train.get('d1'))

audio_test = h5py.File("audio_test.h5","r")
print(list(audio_test.keys()))
audio_test = np.array(audio_test.get('d1'))

text_train_emb = h5py.File("text_train_emb.h5","r")
print(list(text_train_emb.keys()))
text_train_emb = np.array(text_train_emb.get('d1'))

text_test_emb = h5py.File("text_test_emb.h5","r")
print(list(text_test_emb.keys()))
text_test_emb = np.array(text_test_emb.get('d1'))
text_test_emb.shape

from keras.models import load_model
import keras
import tensorflow as tf
from time import time
from keras import layers
from google.colab import files	
from keras.models import load_model
from keras.models import Model,Sequential,Model
from keras.layers import *
from keras.callbacks import TensorBoard
from keras import callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

model_audio_sad = load_model("unimodal baselines/weights/audio/weights_rnn_sad.h5")
model_video_sad = load_model("unimodal baselines/weights/video/weights_rnn_sad.h5")
model_text_sad = load_model("unimodal baselines/weights/text/weights_rnn_sad.h5")


sad_video = Sequential()
sad_audio = Sequential()
sad_text = Sequential()

for layer in model_video_sad.layers[:-2]:   #(typical late fusion)
    sad_video.add(layer)
for layer in sad_video.layers:
    layer.trainable = False
for layer in model_audio_sad.layers[:-2]:   #(typical late fusion)
    sad_audio.add(layer)
for layer in sad_audio.layers:
    layer.trainable = False
for layer in model_text_sad.layers[:-2]:   #(typical late fusion)
    sad_text.add(layer)
for layer in sad_text.layers:
    layer.trainable = False
sad_video.summary()

i1 = Input(shape=(20,35),name='i1')
i2 = Input(shape=(20,74),name='i2')
i3 = Input(shape=(20,300),name='i3')

o1 = sad_video(i1)
o2 = sad_audio(i2)
o3 = sad_text(i3)

def tensor_fuse(out):
	o1 = out[0]     # o1.shape = (None,32)
	o2 = out[1]     # o2.shape = (None,32)
	o3 = out[2]     # o3.shape = (None,32)

	x = tf.expand_dims(o1,1)   # x.shape = (None,1,32)
	y = tf.expand_dims(o2,2)   # y.shape = (None,32,1)
	o12 = tf.reshape(tf.multiply(x,y),shape=[-1,32*32])  # o12.shape = (None,32*32)

	x = tf.expand_dims(o2,1)   # x.shape = (None,1,32)
	y = tf.expand_dims(o3,2)   # y.shape = (None,32,1)
	o23 = tf.reshape(tf.multiply(x,y),shape=[-1,32*32])  # o23.shape = (None,32*32)

	x = tf.expand_dims(o3,1)   # x.shape = (None,1,32)
	y = tf.expand_dims(o1,2)   # y.shape = (None,32,1)
	o31 = tf.reshape(tf.multiply(x,y),shape=[-1,32*32])  # o31.shape = (None,32*32)

	x = tf.expand_dims(o12,2)  # x.shape = (None,32*32,1)
	y = tf.expand_dims(o3,1)   # y.shape = (None,1,32)
	o123 = tf.reshape(tf.multiply(x,y),[-1,32*32*32])  # o123.shape = (None,32*32*32)

	return concatenate([o1,o2,o3,o12,o23,o31,o123])

merged = layers.Lambda(tensor_fuse)([o1,o2,o3])
y = Dense(1024,activation='relu')(merged)
y = Dropout(rate=0.4)(y)
y = Dense(1024,activation='relu')(y)
y = Dense(64,activation='relu')(y)
y = Dense(64,activation='relu')(y)
y = Dense(1,activation='sigmoid')(y)

model = Model(inputs=[i1,i2,i3],outputs=y)
model.compile('adam','binary_crossentropy',metrics=['accuracy'])
model.summary()

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
es = EarlyStopping(monitor='val_loss',mode='min' ,patience=5, min_delta=0.0001,verbose=1)
mcp = ModelCheckpoint("tensor_fusion_sad.h5",monitor='val_loss',verbose=1)

model.fit({'i1':video_train,'i2':audio_train,'i3':text_train_emb},Train_labels_sad,128,epochs=20,validation_split=0.1,callbacks=[es, mcp, tensorboard])
print(model.evaluate({'i1':video_test,'i2':audio_test,'i3':text_test_emb},Test_labels_sad))
files.download('tensor_fusion_sad.h5')