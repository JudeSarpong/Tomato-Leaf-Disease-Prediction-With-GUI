import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator 
from keras.applications.vgg19 import VGG19, preprocess_input 

#Run this python file to save the model as best_model.h5 or use any other name
#Check the path to directory of the dataset. Get Dataset at https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf

train_datagen = ImageDataGenerator(zoom_range =0.5, shear_range=0.3, horizontal_flip = True, preprocessing_function= preprocess_input)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train = train_datagen.flow_from_directory(directory='D:\Comp Science\KnowIt\AllImages\Train', target_size=(256, 256),batch_size=32)
val =  val_datagen.flow_from_directory(directory='D:\Comp Science\KnowIt\AllImages\Test', target_size=(256, 256),batch_size=32)

t_img, label = train.next()


def plotImage(img_arr, label):

  for im , l in zip(img_arr, label):
    plt.figure(figsize=(5,5))
    plt.imshow(im/255)
    plt.show()


plotImage(t_img[:4], label[:4])

from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
import keras

base_model = VGG19(include_top = False, input_shape=(256,256,3), classes =10, classifier_activation= 'softmax')

for layer in base_model.layers:
  layer.trainable = False 

print("\n\nBASE MODEL SUMMARY:")
base_model.summary()

x = Flatten()(base_model.output)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(units = 10, activation = 'softmax')(x)

model = Model(base_model.input, x)


print('\n\nFINAL MODEL SUMMARY:')
model.summary()

model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping
es = EarlyStopping(monitor = 'val_accuracy', min_delta = 0.01, patience = 3, verbose = 1)

mc = ModelCheckpoint(filepath = "best_model.h5", monitor = 'val_accuracy', 
                     min_delta = 0.01, patience = 15, verbose = 1, save_best_only = True )

cb = [es, mc]

#where es, mc and cb represents Earlystopping, Modelcheckpoint and callback respectively



his = model.fit(train, steps_per_epoch = 200, epochs = 100, verbose=1, callbacks = cb, validation_data = val,
                              validation_steps = 16)



