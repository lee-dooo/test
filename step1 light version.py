import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. M3 GPU(Metal) 가속 확인 및 최적화
device = "GPU" if len(tf.config.list_physical_devices('GPU')) > 0 else "CPU"
print(f"학습 장치: {device}")

path = "/Users/leedooo/Image Processing/aerospace_images"
batch_size = 32 # 배치 사이즈를 키워 GPU 효율을 높임
target_size = (128, 128) # 해상도 축소 (224 -> 128)

# 2. 데이터 제너레이터 단순화 (속도를 위해 복잡한 변형 제거)
image_generator = ImageDataGenerator(rescale=1/255., validation_split=0.2)

train_dataset = image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=path,
    shuffle=True,
    target_size=target_size, # 축소된 사이즈 적용
    subset="training",
    class_mode="categorical")

validation_dataset = image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=path,
    shuffle=True,
    target_size=target_size,
    subset="validation",
    class_mode="categorical")

# 3. 모델 변경: VGG19 -> MobileNetV2 (속도 차이가 매우 큽니다)
num_of_categories = 7
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(target_size[0], target_size[1], 3),
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = False # 기본 모델 동결 (학습 시간 단축)

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_of_categories, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 학습 진행
hist = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset, 
    verbose=1
)

# 5. 그래프 시각화 (오타 수정 포함)
plt.figure(figsize=(10, 5))
plt.plot(hist.history["loss"], label="Loss")
plt.plot(hist.history["accuracy"], label="Accuracy")
plt.plot(hist.history["val_loss"], label="Val_loss")
plt.plot(hist.history["val_accuracy"], label="Val_accuracy")
plt.title("Model Accuracy & Loss")
plt.ylabel("Value")
plt.xlabel("Epoch")
plt.legend()
plt.show()