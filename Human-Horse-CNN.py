import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PySimpleGUI as sg
from keras.utils import load_img, img_to_array
import os

train_horse_dir = os.path.join('/Users/muhammadhamzasohail/Desktop/horse-or-human/horses')
train_human_dir = os.path.join('/Users/muhammadhamzasohail/Desktop/horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

path_img= sg.popup_get_file('CNN Based predictor : Please select the image you need to process:')

img = mpimg.imread(path_img)
imgplot = plt.imshow(img)
plt.show()

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) 
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) 
                for fname in train_human_names[pic_index-8:pic_index]]

# for i, img_path in enumerate(next_horse_pix+next_human_pix):
#   # Set up subplot; subplot indices start at 1
#   sp = plt.subplot(nrows, ncols, i + 1)
#   sp.axis('Off') # Don't show axes (or gridlines)

#   img = mpimg.imread(img_path)
# #   plt.imshow(img)

# # plt.show()


model = tf.keras.models.Sequential([

   
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
 
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
  
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])



from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255

path = os.path.join('/Users/muhammadhamzasohail/Desktop/horse-or-human')


train_datagen = ImageDataGenerator(rescale=1/255)


train_generator = train_datagen.flow_from_directory(
        path, 
        target_size=(300, 300),  
        batch_size=128,

        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1)


model.save('model.h5')

img = load_img(path_img, target_size=(300, 300))
x = img_to_array(img)
x /= 255
x = np.expand_dims(x, axis=0)
classes = model.predict(x)
print(classes[0])
result =""
if classes[0]>0.5:
    result = "The image seems to be of a human"
    print("You image prediction based on the CNN model is of a human.")
else:
    result = "You image prediction based on the CNN model is of a horse"
    print("You image prediction based on the CNN model is of a horse.")



import PySimpleGUI as sg

layout = [[sg.Text(result)], [sg.Button("OK")]]

# Create the window
window = sg.Window("Demo", layout)

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "OK" or event == sg.WIN_CLOSED:
        break

window.close()