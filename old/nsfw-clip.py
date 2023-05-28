
import time

import autokeras as ak

import tensorflow as tf




import numpy as np

neutral = np.load ("./neutral/img_emb/img_emb_0.npy")
print(neutral.shape)

porn = np.load ("./porn/img_emb/img_emb_0.npy")
print(porn.shape)

drawings = np.load ("./drawings/img_emb/img_emb_0.npy")
print(drawings.shape)

hentai = np.load ("./hentai/img_emb/img_emb_0.npy")
print(hentai.shape)

sexy = np.load ("./sexy/img_emb/img_emb_0.npy")
print(sexy.shape)



x_t =np.concatenate((porn,sexy),axis = 0)
x_t =np.concatenate((x_t,hentai),axis = 0)
nsfw_t_len=x_t.shape[0]
print(nsfw_t_len)
x_t =np.concatenate((x_t,neutral),axis = 0)
x_t =np.concatenate((x_t,drawings),axis = 0)
y_t = np.zeros(x_t.shape[0], dtype = np.uint8)
sfw_t_len=x_t.shape[0] - nsfw_t_len
print(sfw_t_len)

for i in range(nsfw_t_len):
  y_t[i]=1
from sklearn.utils import shuffle
x_train, y_train = shuffle(x_t, y_t)


print(y_t)
print(y_train)







x_train = x_train.astype(float) #[100:-100]
y_train = y_train.astype(int)#[100:-100]


#x_test = x_test.astype(float) #[100:-100]
#y_test = y_test.astype(int)#[100:-100]


# It tries 10 different models.
clf = ak.StructuredDataClassifier(overwrite=True, max_trials=5)

# Feed the structured data classifier with training data.
clf.fit(x_train, y_train, epochs=10, validation_split=0.1)

model = clf.export_model()
model.summary()




model.save("clip_autokeras_nsfw")