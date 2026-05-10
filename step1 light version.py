import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. M3 GPU(Metal) 가속 확인: 현재 컴퓨터가 GPU를 쓸 수 있는지 체크합니다.
device = "GPU" if len(tf.config.list_physical_devices('GPU')) > 0 else "CPU"
print(f"학습 장치: {device}")

# 경로 및 하이퍼파라미터 설정
path = "/Users/leedooo/Image Processing/aerospace_images"
batch_size = 32          # 한 번에 학습할 이미지 묶음 크기
target_size = (128, 128) # 모델에 들어갈 이미지 크기 (작을수록 학습이 빠름)

# 2. 데이터 전처리기 설정: 사진을 0~1 사이 숫자로 바꾸고, 20%는 검증용으로 분리하겠다는 규칙입니다.
image_generator = ImageDataGenerator(rescale=1/255., validation_split=0.2)

# 실제 폴더에서 학습용 데이터를 가져옵니다.
train_dataset = image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=path,
    shuffle=True,          # 데이터를 섞어서 학습 효과를 높임
    target_size=target_size,
    subset="training",     # 80%에 해당하는 학습 데이터
    class_mode="categorical") # 다중 분류(7개 클래스) 설정

# 실제 폴더에서 검증용 데이터를 가져옵니다.
validation_dataset = image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=path,
    shuffle=True,
    target_size=target_size,
    subset="validation",   # 20%에 해당하는 검증 데이터
    class_mode="categorical")

# 3. 모델 구축 (MobileNetV2 사용)
num_of_categories = 7 # 분류할 대상의 개수

# 베이스 모델: 이미 학습된 지능(ImageNet 지식)을 가진 거대한 신경망을 가져옵니다.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(target_size[0], target_size[1], 3),
    include_top=False, # 모델의 마지막 출력층(1000개 분류)은 빼고 가져옵니다.
    weights='imagenet' # 이미지넷에서 미리 공부한 '가중치'를 사용합니다.
)
base_model.trainable = False # 이미 똑똑한 베이스 모델의 지식은 고정(동결)합니다.

# 내 문제(7개 분류)에 맞게 새로운 층을 쌓습니다.
model = tf.keras.Sequential([
    base_model,                                     # 위에서 가져온 베이스 모델
    tf.keras.layers.GlobalAveragePooling2D(),       # 복잡한 특징을 평균내어 단순화
    tf.keras.layers.Dense(num_of_categories, activation='softmax') # 최종 7개 확률 출력
])

# 공부 방법 결정: Adam 방식의 최적화 도구와 오차 계산법을 설정합니다.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 실제 학습 진행 (훈련 데이터로 공부하고 검증 데이터로 시험 봄)
hist = model.fit(
    train_dataset,
    epochs=3, # 전체 데이터를 3번 반복해서 공부함
    validation_data=validation_dataset, 
    verbose=1 # 학습 과정을 화면에 보여줌
)

# 5. 그래프 시각화: 학습이 잘 되었는지(오차가 줄고 정확도가 오르는지) 확인합니다.
plt.figure(figsize=(10, 5))
plt.plot(hist.history["loss"], label="Loss")
plt.plot(hist.history["accuracy"], label="Accuracy")
plt.plot(hist.history["val_loss"], label="Val_loss")
plt.plot(hist.history["val_accuracy"], label="Val_accuracy")
plt.title("Model Accuracy & Loss")
plt.legend()
plt.show()

# 6. 모델 저장 프로세스
import os
save_dir = "/Users/leedooo/Image Processing"
if not os.path.exists(save_dir):
    os.makedirs(save_dir) # 폴더가 없으면 새로 만듦

# 최신 형식인 .keras로 저장 경로 설정
save_path = os.path.join(save_dir, "air_vgg19_model.keras") 

try:
    model.save(save_path) # 모델의 구조와 학습된 지능(가중치)을 파일로 저장
    print(f"✅ 모델이 성공적으로 저장되었습니다: {save_path}")
    
    if os.path.exists(save_path):
        print(f"🔎 파일 확인 완료! 용량: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
except Exception as e:
    print(f"❌ 저장 실패! 에러 내용: {e}")