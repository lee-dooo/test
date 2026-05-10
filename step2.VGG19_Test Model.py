import tensorflow as tf
import cv2
import os
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
import numpy as np

IMAGE_SIZE = 128

ImagesFolder = "/Users/leedooo/Image Processing/aerospace_images"
# .DS_Store 같은 숨김 파일을 제외하고 클래스 목록을 가져오는 것이 안전합니다.
CLASSES = sorted([f for f in os.listdir(ImagesFolder) if not f.startswith('.')])
print("클래스 목록:", CLASSES)

best_model_file = "/Users/leedooo/Image Processing/air-vgg19.h5"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())

def parpareImage(pathForImage) :
    image = load_img(pathForImage, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255.
    return imgResult

testImagePth = "/Users/leedooo/Image Processing/aerospace_images/dirigible/10121.jpg"
 
img = cv2.imread(testImagePth)
imgForModel = parpareImage(testImagePth)

# 예측 수행
resultArray = model.predict(imgForModel, verbose=1)
answer = np.argmax(resultArray, axis=1)

print(answer)

index = answer[0]
className = CLASSES[index]

print("The predicted class is : " + className)
