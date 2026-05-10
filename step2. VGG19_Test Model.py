import tensorflow as tf
import cv2
import os
from keras.utils import load_img, img_to_array
import numpy as np

# 1. 모델이 학습할 때 약속했던 이미지 크기 (매우 중요!)
IMAGE_SIZE = 128

# 2. 정답 레이블(클래스 이름) 목록 만들기
ImagesFolder = "/Users/leedooo/Image Processing/aerospace_images"
# 폴더 내 숨김파일을 제외하고 알파벳 순서(sorted)로 가져와 학습 때와 순서를 맞춥니다.
CLASSES = sorted([f for f in os.listdir(ImagesFolder) if not f.startswith('.')])
print("클래스 목록:", CLASSES)

# 3. 저장된 모델 파일 불러오기
best_model_file = "/Users/leedooo/Image Processing/air-vgg19.h5"
model = tf.keras.models.load_model(best_model_file)
print(model.summary()) # 불러온 모델의 구조를 다시 확인

# 4. 이미지 전처리 함수: 원본 사진을 모델이 이해할 수 있는 형식으로 변환
def parpareImage(pathForImage) :
    # 사진을 불러오면서 모델 크기(128x128)에 맞게 강제로 조절합니다.
    image = load_img(pathForImage, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    # 이미지를 숫자 배열(Numpy Array)로 바꿉니다.
    imgResult = img_to_array(image)
    # 모델은 묶음(Batch) 단위로 처리하므로, (1, 128, 128, 3)처럼 앞차원을 하나 추가합니다.
    imgResult = np.expand_dims(imgResult, axis=0)
    # 픽셀 값을 0~1 사이로 정규화합니다 (학습 때와 동일하게).
    imgResult = imgResult / 255.
    return imgResult

# 5. 테스트할 실제 이미지 경로
testImagePth = "/Users/leedooo/Image Processing/aerospace_images/dirigible/10121.jpg"
 
# OpenCV로 이미지를 읽어옵니다 (시각화 등에 활용 가능).
img = cv2.imread(testImagePth)
# 위에서 만든 함수를 통해 모델 전용 데이터로 가공합니다.
imgForModel = parpareImage(testImagePth)

# 6. 예측 수행: 모델에게 사진을 보여주고 결과값을 받습니다.
# resultArray에는 7개 카테고리에 대한 각각의 확률값이 들어있습니다.
resultArray = model.predict(imgForModel, verbose=1)

# 7. 결과 해석: 확률이 가장 높은 인덱스(번호)를 찾습니다.
answer = np.argmax(resultArray, axis=1)
print("예측 번호:", answer)

# 번호에 해당하는 클래스 이름을 가져옵니다.
index = answer[0]
className = CLASSES[index]

# 8. 최종 결과 출력
print("The predicted class is : " + className)