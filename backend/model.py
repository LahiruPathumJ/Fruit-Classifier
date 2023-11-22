import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

#First we need to create the Image Augmantation
traning_datagen=ImageDataGenerator(shear_range=0.2,zoom_range=0.2,horizontal_flip='True')
test_datagen=ImageDataGenerator()

#creating the test & train sets
traning_set=traning_datagen.flow_from_directory('../../dataset/train',batch_size=60,target_size=(64,64),class_mode='categorical')
test_set=test_datagen.flow_from_directory('../../dataset/test',batch_size=60,target_size=(64,64),class_mode='categorical')
print("Loaded dataset.......")



model=tf.keras.Sequential()
#first layer
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),input_shape=[64,64,3],activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
#second
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
#third
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
#flatten
model.add(tf.keras.layers.Flatten())
#adding the nural net
model.add(tf.keras.layers.Dense(units=256,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(units=521,activation='relu'))
model.add(tf.keras.layers.Dense(units=6,activation='softmax'))
model.summary()
print("Created model........")


#compling
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x=traning_set,validation_data=test_set,epochs=30, batch_size=32)
print("Compiled model.............")


#ploting the accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
#ploting the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()



#save your model and download it or add drive path
model.save('Fresh_Rotten_fruis.h5')#saving the weights of nural network
model_json = model.to_json()       #Saving in json form
with open("Fresh_Rotten_Fruits.json", "w") as json_file:
    json_file.write(model_json)
print('Saved the model..............')



import numpy as np
from keras.preprocessing import image
np.loadtxt
classes = ['Fresh Apple','Fresh Banana','Fresh Orange','Rotten Apple','Rotten Banana','Rotten Orange']#creating the class labels
test_image = image.load_img('../../dataset/train/freshapples/rotated_by_15_Screen Shot 2018-06-08 at 5.03.34 PM.png', target_size = (64, 64))#upload your image
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(np.argmax(result))
result1=result[0]
for i in range(6):
  if result1[i] == 1.:
    break
prediction = classes[i]
print("Predicted output: ")
print(prediction)


import pickle

# Save the trained model to a .pkl file
with open('fresh_rotten_fruits.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Created dump.................")


import joblib

# Save the trained model to a .pkl file using joblib
joblib.dump(model, 'fresh_rotten_fruits.pkl')