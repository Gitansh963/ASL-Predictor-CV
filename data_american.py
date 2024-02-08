
import numpy as np 
import pandas as pd
import cv2
import os
from tqdm import tqdm


base_path = 'American/'
classes = os.listdir(base_path)
filepaths = []
labels = []
for d in classes:
    flist = os.listdir(base_path + d)
    for f in flist:
        fpath = os.path.join(base_path + d + '/' + f)
        filepaths.append(fpath)
        labels.append(d)
print ('filepaths: ', len(filepaths), '   labels: ', len(labels))


series1=pd.Series(filepaths, name='file_paths')
series2=pd.Series(labels, name='labels')
df=pd.concat([series1,series2], axis=1)

df.head(1)

df=pd.DataFrame(np.array(df).reshape(142261,2), columns = ['file_paths', 'labels'])
print(df['labels'].value_counts())


balanced_df = df.groupby('labels').apply(lambda x: x.sample(1500)).reset_index(drop=True)
print(balanced_df['labels'].value_counts())


image_paths = balanced_df['file_paths'].tolist()
images = []
images_flatten = []

for path in image_paths:
    print(path)
    image = cv2.imread(path)
    image_reshaped = cv2.resize(image,(200,200))
    image_flatten = np.array(image_reshaped).flatten()
    images.append(image_reshaped)
    images_flatten.append(image_flatten)

images_array = np.array(images)
images_array.shape

dfs = pd.DataFrame(index=range(540), columns=['image_array'])

for i, image_array in enumerate(images_array):
    dfs.at[i, 'image_array'] = image_array

images_array.shape

labels = np.array(balanced_df['labels'])

# numpy = 1.26.2
np.save('images_american.npy', images_array)
np.save('labels_american.npy', labels)


df.to_csv('df.csv', index=False)
balanced_df.to_csv('balanced_df.csv', index=False)
