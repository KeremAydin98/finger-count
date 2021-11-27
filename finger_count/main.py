import os
from matplotlib.image import imread
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import cv2
from tensorflow.keras.callbacks import EarlyStopping

data_dir = 'C:\\Users\\KEREM\\Desktop\\gray_finger_count'

#print(os.listdir(data_dir))

train_dir = data_dir + "\\train"
test_dir = data_dir + "\\test"

#print(os.listdir(test_dir))

#print(os.listdir(train_dir + "\\1")[0])

cell = train_dir + "\\1\\" + "one (13).png"

#print(imread(cell).shape)

#print(len(os.listdir(train_dir + "\\1")))
#print(len(os.listdir(test_dir + "\\1")))

d1=[]
d2=[]
for i in [1,2,3,4,5]:
    for image_file in os.listdir(train_dir + f"\\{i}"):
        img = imread(train_dir + f"\\{i}\\" + image_file)
        dim1, dim2 = img.shape
        d1.append(dim1)
        d2.append(dim2)

print(np.mean(d1))
print(np.mean(d2))

image_shape = (300,250,1)

image_gen = ImageDataGenerator(rotation_range=30,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               fill_mode='nearest')


train_image_gen = image_gen.flow_from_directory(train_dir,
                                                target_size=image_shape[:2],
                                                color_mode='grayscale',
                                                batch_size=8,
                                                class_mode='categorical')

test_image_gen = image_gen.flow_from_directory(test_dir,
                                                target_size=image_shape[:2],
                                                color_mode='grayscale',
                                                batch_size=8,
                                                class_mode='categorical',
                                                shuffle=False)
model = Sequential()

model.add(Conv2D(filters=32,input_shape=image_shape,
                 kernel_size=(4,4),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,input_shape=image_shape,
                 kernel_size=(4,4),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,input_shape=image_shape,
                 kernel_size=(4,4),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,input_shape=image_shape,
                 kernel_size=(4,4),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(5,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

early_stop=EarlyStopping(monitor='val_loss',mode='min',patience=2)

results = model.fit_generator(train_image_gen,epochs=20,validation_data=test_image_gen,callbacks=[early_stop])

predictions = model.predict_generator(test_image_gen)

predictions = np.argmax(predictions, axis=1)


print(classification_report(test_image_gen.classes,predictions))

print(predictions)

model.save('my_model.h5')





