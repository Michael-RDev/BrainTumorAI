import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.utils import to_categorical

brain_yes = "F:/BrainTumor/data/yes/"
brain_no = "F:/BrainTumor/data/no/"

def getImgs(folder_path:str):
    full_paths = []
    for images in tqdm(os.listdir(folder_path),desc="Doing: "):
        fullPath = os.path.join(folder_path, images)
        imgs = cv2.imread(fullPath)
        imgs = cv2.resize(imgs, (50,50))
        full_paths.append(imgs)
    return full_paths

CancerYes = getImgs(brain_yes)
CancerNo = getImgs(brain_no)

labels_yes = [1] * len(CancerYes)
labels_no = [0] * len(CancerNo)

images = np.array(CancerNo + CancerYes)
labels = np.array(labels_no + labels_yes)

X_train, X_test, y_train, y_test = train_test_split(images, labels, random_state=42, test_size=0.2)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)