import sys

import numpy as np
import struct as st
import gzip
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
def read_data(filename):
    file = gzip.open(filename,'rb')
    file.seek(0)
    magic_number = st.unpack('>4B',file.read(4)) # read magic number
    number_of_images = st.unpack('>I',file.read(4))[0] # read number of images
    number_of_rows = st.unpack('>I',file.read(4))[0] #read number of rows
    number_of_columns = st.unpack('>I',file.read(4))[0] #read number of column
    with gzip.open(filename) as file_stream:
        file_stream.read(16)
        buf = file_stream.read(number_of_rows * number_of_columns * number_of_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(number_of_images, number_of_rows,number_of_columns)
        return data
    
def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):    
    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded


#Check program arguments
if ((len(sys.argv))!=3):
    print("Usage is: python autoencoder.py -d <dataset>")
    sys.exit()
if sys.argv[1] == "-d":
	print("You are correct!")
else:
    print("Usage is: python autoencoder.py -d <dataset>")
    sys.exit()
    
while 1:
    #Read Hyper parameters    
    batch_size = int(input("Please enter batch size: "))
    epochs = int(input("Please enter epochs number: "))
    inChannel = 1
    x, y = 28, 28
    input_img = Input(shape = (x, y, inChannel))

    #Read data file
    train_data = read_data(sys.argv[2])
    train_data = train_data.reshape(-1, 28,28, 1)
    train_data.shape
    train_data = train_data / np.max(train_data)
    
    #Split dataset to train and validation sets
    train_set_X,validation_set_X,train_ground,validation_ground = train_test_split(train_data,train_data,test_size=0.2,random_state=13)

    #Model Complilation and Training
    autoencoder = Model(input_img, decoder(encoder(input_img)))
    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

    autoencoder_train = autoencoder.fit(train_set_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(validation_set_X, validation_ground))

    Ans = input("Choose: \n 1)TYPE \"AGAIN\" to set new hyperparameters and retrain algorithm \n 2)TYPE \"PLOT\" to see loss graph \n 3)TYPE \"SAVE\" to save the model\n 4)TYPE \"EXIT\" to stop program\n")
    
    if Ans=="PLOT":
        loss = autoencoder_train.history['loss']
        val_loss = autoencoder_train.history['val_loss']
        epochs = range(epochs)
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        sys.exit()
    if Ans=="SAVE":
        autoencoder.save('AutoencoderModel.h5')
        sys.exit()
    if Ans=="EXIT":
        sys.exit()
