### Add lines to import modules as needed
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import Input, layers
from PIL import Image
print(tf.__version__)
## 

def build_model1():
  model = tf.keras.Sequential([
    Input(shape=(32,32,3)),
    layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
    layers.BatchNormalization(),
    
    layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
    layers.BatchNormalization(),
    
    layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
    layers.BatchNormalization(),

    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),

    layers.MaxPooling2D(pool_size=(4,4), strides=(4,4)),    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10)
  ])
  return model

def build_model2():
  model = tf.keras.Sequential([
    Input(shape=(32,32,3)),
    layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
    layers.BatchNormalization(),
    
    layers.SeparableConv2D(64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
    layers.BatchNormalization(),
    
    layers.SeparableConv2D(128, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
    layers.BatchNormalization(),

    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),

    layers.MaxPooling2D(pool_size=(4,4), strides=(4,4)),    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10)
  ])
  return model

def build_model3():
  inputs = Input(shape=(32,32,3))
  x = layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.3)(x)

  # First 2 conv blocks
  y = layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same')(x)
  y = layers.BatchNormalization()(y)
  y = layers.Dropout(0.3)(y)
  y = layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same')(y)
  y = layers.BatchNormalization()(y)
  y = layers.Dropout(0.3)(y)

  # Skip connection
  x = layers.Conv2D(128, kernel_size=(1,1), strides=(4,4), padding='same')(x)
  x = layers.Add()([x,y])

  # Second 2 conv blocks
  y = layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same')(x)
  y = layers.BatchNormalization()(y)
  y = layers.Dropout(0.3)(y)
  y = layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same')(y)
  y = layers.BatchNormalization()(y)
  y = layers.Dropout(0.3)(y)

  # Skip connection
  x = layers.Add()([x,y])

  # Third 2 conv blocks
  y = layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same')(x)
  y = layers.BatchNormalization()(y)
  y = layers.Dropout(0.3)(y)
  y = layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same')(y)
  y = layers.BatchNormalization()(y)
  y = layers.Dropout(0.3)(y)

  # Skip connection
  x = layers.Add()([x,y])

  x = layers.MaxPooling2D(pool_size=(4,4), strides=(4,4))(x)
  x = layers.Flatten()(x)
  x = layers.Dense(128, activation='relu')(x)
  x = layers.BatchNormalization()(x)
  outputs = layers.Dense(10)(x)

  model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
  return model

def build_model50k():
  model = tf.keras.Sequential([
    Input(shape=(32,32,3)),
    layers.Conv2D(16, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Conv2D(16, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Conv2D(16, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.3),

    layers.Conv2D(32, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Conv2D(32, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.3),

    layers.Conv2D(64, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(10)
  ])
  return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':

  tf.random.set_seed(777)
  np.random.seed(777)

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

  val_frac = 0.1
  num_val_samples = int(len(train_images)*val_frac)
  val_idxs = np.random.choice(np.arange(len(train_images)), size=num_val_samples, replace=False)
  trn_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)
  val_images = train_images[val_idxs, :,:,:]
  train_images = train_images[trn_idxs, :,:,:]

  val_labels = train_labels[val_idxs]
  train_labels = train_labels[trn_idxs]

  train_labels = train_labels.squeeze()
  test_labels = test_labels.squeeze()
  val_labels = val_labels.squeeze()

  input_shape  = train_images.shape[1:]
  train_images = train_images / 255.0
  test_images = test_images / 255.0
  val_images = val_images / 255.0

  ########################################
  ## Build and train model 1
  model1 = build_model1()
  model1.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  model1.summary()
  train_hist = model1.fit(train_images, train_labels, 
                          validation_data=(val_images, val_labels),
                          epochs=50)
  test_loss, test_acc = model1.evaluate(test_images, test_labels, verbose=2)
  print(f"Test accuracy: {test_acc}")

  img = Image.open('simple-convolutional-neural-networks-charlie-51135/test_image_dog.jpg')
  img_array = np.array(img)
  img_array = img_array / 255.0

  prediction = model1.predict(np.expand_dims(img_array, axis=0)) # add dummy batch dim
  predicted_class = np.argmax(prediction)
  class_probabilities = tf.nn.softmax(prediction[0])

  print(f"Predicted Class: {class_names[predicted_class]}")
  print("Class Probabilities:")
  for i, prob in enumerate(class_probabilities):
      print(f"Class {class_names[i]}: {prob}")

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  model2.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  model2.summary()
  train_hist = model2.fit(train_images, train_labels, 
                          validation_data=(val_images, val_labels),
                          epochs=50)
  test_loss, test_acc = model2.evaluate(test_images, test_labels, verbose=2)
  print(f"Test accuracy: {test_acc}")
  
  ## Repeat for model 3 and your best sub-50k params model
  
  model3 = build_model3()
  model3.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  model3.summary()
  train_hist = model3.fit(train_images, train_labels, 
                          validation_data=(val_images, val_labels),
                          epochs=50)
  test_loss, test_acc = model3.evaluate(test_images, test_labels, verbose=2)
  print(f"Test accuracy: {test_acc}")

  model50k = build_model50k()
  model50k.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  model50k.summary()
  train_hist = model50k.fit(train_images, train_labels, 
                          validation_data=(val_images, val_labels),
                          epochs=50)
  test_loss, test_acc = model50k.evaluate(test_images,  test_labels, verbose=2)
  print(f"Test accuracy: {test_acc}")
  model50k.save("best_model.h5")