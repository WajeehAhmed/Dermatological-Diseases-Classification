# -*- coding: utf-8 -*-
# Author Wajeeh Ahmed
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
%matplitlib inline

mobile = tf.keras.applications.mobilenet.MobileNet()

os.chdir('E:/Stay Away You!/SkinDiseases/skin_dataset')



train_path = 'E:/Stay Away You!/SkinDiseases/skin_dataset/train'
test_path = 'E:/Stay Away You!/SkinDiseases/skin_dataset/test'
valid_path = 'E:/Stay Away You!/SkinDiseases/skin_dataset/valid'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory= train_path,target_size=(224,224),batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory= test_path,target_size=(224,224),batch_size=10,shuffle=False)
validation_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory= valid_path,target_size=(224,224),batch_size=10)
mobile.summary()
# Getting all layers upto -6 layer from bottom

x = mobile.layers[-6].output
output = Dense(units=3,activation = 'softmax')(x)

model = Model(inputs=mobile.input,outputs = output)
        
for layer in model.layers[:-23]:
    layer.trainable=False
    
model.summary()


model.compile(optimizer=Adam(lr = 0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_batches,validation_data = validation_batches,epochs=10,verbose = 2)
model.save("model.h5")

#Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#Loss Graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#confusion matrix plot
test_labels = test_batches.classes
predictions = model.predict(x = test_batches,verbose = 0)
cm = confusion_matrix( y_true = test_labels, y_pred = predictions.argmax(axis=1))
cm_plot_labels = ['Acne','Eczema','Melanoma']
plot_confusion_matrix(cm= cm,classes= cm_plot_labels,title = 'Confusion Metrix')

model = load_model('Dermo_Classify')

#for on image prediction
def prepare_image(file):
    img_path = 'C:\\Users\\Wajeeh Ahmed\\Downloads\\'
    img = image.load_img(img_path + file,target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array_expand_dimension = np.expand_dims(img_array,axis = 0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expand_dimension)    

from IPython.display import Image
Image(filename = 'C:\\Users\\Wajeeh Ahmed\\Downloads\\acne-severe.jpg',width= 300,height=200)

preprocessed_img = prepare_image('acne-severe.jpg')
predictions = model.predict(preprocessed_img)
value =  np.amax(predictions)
index = np.where(predictions == np.amax(predictions))
print(index)   



