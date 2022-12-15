import cv2
import numpy as np
from scipy.cluster.vq import *
from sklearn.preprocessing import normalize
import os
from PIL import Image 

sift = cv2.SIFT_create(1000)
K = 256

centers = np.load("centers.npy")
img_feats = np.load("imfeats.npy")
img_list = np.load("imgmap.npy")


def test(im):
    # img = Image.fromarray(im).save("img.jpg")
    # img = cv2.imread("img.jpg")
    # colored_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    _ , des = sift.detectAndCompute(im,None)
    test_feats = np.zeros(K)
    closest, _ = vq(des, centers)
    np.add.at(test_feats, closest, 1)
    test_feats = normalize(test_feats.reshape(1,-1))[0]
    TOP_FIVE = [img_list[x] for x in np.argsort(np.dot(test_feats,img_feats.T)).tolist()[::-1][:5]]
    return [cv2.cvtColor(cv2.imread(f"database/{i}") , cv2.COLOR_BGR2RGB) for i in TOP_FIVE]
