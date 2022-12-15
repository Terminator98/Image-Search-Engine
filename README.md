# Image-Search-Engine
## It was a college project in Image Processing Course

### Methods used in this project:
- `BOW` (Bag Of Words)
- `SIFT` Algorithm (Feature Extraction Algorithm)
- `KMeans`
- `Gradio`

## How To Use:
#### Step 1 : Setting up the K of `KMeans` and no. of features of `SIFT` and reading images from database folder
``` python
K = 256
feat_list = []
img_list = []
sift = cv2.SIFT_create(1000)
impath = os.listdir("database/")
```

#### Step 2 : Starting the `SIFT` Algorithm then `KMeans`
``` python 
for i in impath:
    Path = "database/" + i
    img = cv2.imread(Path)
    colored_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    _ , des = sift.detectAndCompute(colored_img,None)
    feat_list.append(des)
    img_list.append(i)

centers, _ = kmeans(np.vstack(feat_list), K, 5)
```

#### Step 3 : Calculate the closest distance for each image so that all images become indexed
``` python
img_feats = np.zeros((len(feat_list), K), "float32")
for i in range(len(feat_list)):
	closest , _ = vq(feat_list[i], centers)
	np.add.at(img_feats[i], closest, 1)
  
img_feats = normalize(img_feats)
```

#### Step 4 : Save the weights in .npy files so that the retrieval process becomes matter of 1 ~ 2 seconds
``` python
np.save("imfeats.npy" , img_feats)
np.save("centers.npy" , centers)
np.save("imgmap.npy" , img_list)
```

#### Step 5 : Run Gradio Server 
``` python
import gradio as gr
import test

demo = gr.Interface(test.test ,
 gr.Image(type = "numpy") ,
 [gr.Image(shape=(400,400)) for i in range(5)])
demo.launch()
```
#### TO DO List:
- [ ] Make the user have the flexibility to retrieve based on some features (Color, Texture , etc.)
