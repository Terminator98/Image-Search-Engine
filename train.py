import cv2
import numpy as np
from scipy.cluster.vq import *
from sklearn.preprocessing import normalize
import os


K = 256

feat_list = []
img_list = []
sift = cv2.SIFT_create(1000)

print("Extracting SIFT Features...")

impath = os.listdir("database/")

# Training
for i in impath:
    Path = "database/" + i
    img = cv2.imread(Path)
    colored_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    _ , des = sift.detectAndCompute(colored_img,None)
    feat_list.append(des)
    img_list.append(i)

print("Feature Extraction is Done")
print("K Means Starting...")

centers, _ = kmeans(np.vstack(feat_list), K, 5)

print("K Means Completed")
img_feats = np.zeros((len(feat_list), K), "float32")
for i in range(len(feat_list)):
	closest , _ = vq(feat_list[i], centers)
	np.add.at(img_feats[i], closest, 1)


img_feats = normalize(img_feats)

print("Training Done .....")

# Saving the weights so that no need to train each time of retrieval 
np.save("imfeats.npy" , img_feats)
np.save("centers.npy" , centers)
np.save("imgmap.npy" , img_list)