from __future__ import print_function
import numpy as np
from PIL import Image



def get_data_generator(df, indices, for_training, image_path, patches=17,batch_size=1):
    images, box_coordinates, landmarks = [], [],[]
    while True:
        idx = 0
        for i in range(indices//patches):
            images_seq, landmarks_seq = [], []
            for p in range(patches):


                r = df.iloc[idx]
                file, box_coordinates,lmark= r['image_id'], r[['x_min','y_min','x_max','y_max']].values,r[['x1','y1','x2','y2','x3','y3','x4','y4']].values
                im = Image.open(image_path + file)
                im=np.array(im)
                im= np.delete(im,[1,2],axis=2)
                #print (lmark)
                #im = np.expand_dims(im, axis=-1)
                #print ('the shape of image is ', im.shape)
                im = np.array(im) / 255.0
                images_seq.append(im)
                landmarks_seq.append(lmark)
                idx+=1
            images.append(np.array(images_seq))

            landmarks.append(np.array(landmarks_seq))
            if len(images) >= batch_size:
                print (np.array(images).shape)
                #print (np.array(landmarks).shape)
                yield np.array(images),np.array(landmarks)
                images, box_coordinates, landmarks = [], [], []
        if not for_training:
            break

