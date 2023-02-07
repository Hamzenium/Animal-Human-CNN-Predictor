import tkinter as tk
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tkinter import messagebox
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model.h5')

def classify_image():
    file_path = filedialog.askopenfilename(defaultextension=".*", filetypes=[("All Files", ".*")])
    try:
          img = load_img(file_path, target_size=(300, 300))
          img1 = mpimg.imread(file_path)
          imgplot = plt.imshow(img1)
          plt.show()
          x = img_to_array(img)
          x /= 255
          x = np.expand_dims(x, axis=0)
          classes = model.predict(x)
          percent = classes[0][0]
          percent = percent*100
          if classes[0]>0.5:
            result = "The image seems to be of a human"
          else:
            result = "You image prediction based on the CNN model is of a horse"    
          label.config(text=f"Prediction: {result}, Accuracy: {int(percent)}%")
    except:
        messagebox.showerror("Error", "Failed to classify image.")

root = tk.Tk()
root.title("CNN Image Classification")
root.geometry("400x200+100+100")

button = tk.Button(root, text="Classify Image", command=classify_image)
button.pack()

label = tk.Label(root)
label.pack()

root.mainloop()
