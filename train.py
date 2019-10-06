import numpy as np
import os
import argparse
from keras.models import *
from keras.optimizers import *
import pandas as pd
from all_utils import get_data_generator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import callbacks
from LSTM import Densenet_LSTM

'''
Command line arguments:
    "-i" : path to data folder (train, validation set)  
    "-o" : output path where model and logs will be saved 
    Data Folder Structure:
    
    Main Project Folder
    |-- Data
    |  |-- train_patch_images
    |  |-- val_patch_images
    |  |-- train_patches_gnd.csv
    |  |-- val_patches_gnd.cdv
    |-- outputs
    
    Default output path is outputs. Provide a new path in command line to save model to another directory
'''

if __name__ == "__main__":

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=False, help="path to Dataset Folder")
    ap.add_argument("-o", "--output", required=False, help="output path for saving model")
    args = vars(ap.parse_args())

    input_path = args["input"]
    out_path = args["output"]

    if input_path:
        data_directory= input_path
    else:
        data_directory= 'Data'

    if not out_path:
        out_path='outputs'

    # load training and validation dataset
    df_train_path = os.path.join(data_directory,'train_patches_gnd.csv')
    df_val_path =os.path.join(data_directory,'val_patches_gnd.csv')
    train_image_path = os.path.join(data_directory,'train_patch_images/')
    val_image_path = os.path.join(data_directory,'val_patch_images/')

    df_train = pd.read_csv(df_train_path)
    df_val = pd.read_csv (df_val_path)

    train_idx = len(df_train)
    val_idx= (len(df_val))

    batch_size = 6

    # call custom data generator, validation is performed after each training epoch
    train_gen = get_data_generator(df_train, train_idx, for_training=True,image_path=train_image_path, patches= 17,batch_size=batch_size)
    valid_gen = get_data_generator(df_val, val_idx, for_training=True, image_path=val_image_path, patches=17, batch_size=batch_size)


    if not os.path.exists(out_path):
        os.mkdir(out_path)


    if 'model-last_epoch.h5' in os.listdir(out_path):
        print ('last model loaded')
        model= load_model(out_path+'/model-last_epoch.h5')

    else:
        print('created a new model instead')
        model = Densenet_LSTM()

    # training parameters
    adamOpt = Adam(lr=0.0001)
    reduceLearningRate = 0.5
    model.compile(loss= 'mean_squared_error', optimizer=adamOpt, metrics=['mae', 'mse'])
    model.summary(line_length=110)

    # logs and callbacks
    checkpoint_filepath = out_path, '/model-{epoch:03d}.h5'
    log_filename = out_path + '/landmarks' +'_results.csv'
    csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)
    checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [csv_log, checkpoint]
    callbacks_list.append(ReduceLROnPlateau(factor=reduceLearningRate, patience=200,verbose=True))
    callbacks_list.append(EarlyStopping(verbose=True, patience=200))
    callbacks_list.append(TensorBoard(log_dir=out_path+'/logs/'))
    
    
        

    # train the model
    history = model.fit_generator(train_gen,
                        steps_per_epoch=train_idx//batch_size,
                        epochs=1000,
                        callbacks=callbacks_list,
                        validation_data=valid_gen,
                        validation_steps= val_idx//batch_size)
