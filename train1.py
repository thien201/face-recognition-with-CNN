import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Tạo generator để tăng cường dữ liệu
data_gen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Đường dẫn tới thư mục dataset
dataset_path = 'C:/Users/nguyenvanthien/Desktop/Code/dataset/'

data = []
labels = []
label_names = []

# Lặp qua các thư mục con trong dataset_path
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        label_names.append(folder_name)
        # Lặp qua các file ảnh trong từng thư mục của cầu thủ
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            # Đọc ảnh và xử lý
            Img = cv2.imread(image_path)
            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
            Img = cv2.resize(Img, (100, 100))
            Img = np.array(Img)
            data.append(Img)
            labels.append(folder_name)

# Chuyển đổi labels thành dạng số nguyên
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Chuyển đổi data và labels thành numpy arrays
data = np.array(data)
labels = np.array(labels)
data = data.reshape((data.shape[0], 100, 100, 1))

# Chuẩn hóa dữ liệu bằng cách chia cho 255 (giá trị pixel tối đa)
X_train = data / 255.0
Y_train = labels

# Xây dựng mô hình
model = Sequential()
shape = (100, 100, 1)
model.add(Conv2D(32, (3, 3), padding="same", input_shape=shape))
model.add(Activation("relu"))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
# Số lớp đầu ra phải bằng số lượng folder (đại diện cho từng cầu thủ)
model.add(Dense(len(label_names)))  # Sửa lại số lớp đầu ra phù hợp
model.add(Activation("softmax"))

# Compile và train mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()  # Xem lại summary để đảm bảo mô hình được cấu hình đúng
# Train mô hình sử dụng data augmentation
print("Start training with augmentation")
history = model.fit(
    data_gen.flow(X_train, Y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=100
)
# Lưu mô hình sau khi train
model.save("khuonmat_augmented.h5")

# Lưu nhãn ra file để sau này sử dụng
with open('labels_augmented.dat', 'wb') as file:
    pickle.dump(lb, file)

# Vẽ đồ thị loss và accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), history.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("training_plot.png")
plt.show()