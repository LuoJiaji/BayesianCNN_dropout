import cv2 
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, Dropout, Layer,Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop, SGD
from keras import callbacks, optimizers
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model,load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

input_shape = (28,28,1)
input_data = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_data)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
x = Flatten(name='flatten')(x)
x = Dense(128, activation='relu', name='fc1')(x)
#x = Lambda(lambda x: K.dropout(x, 0.5))(x)
#x = Dropout(0.5)(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Lambda(lambda x: K.dropout(x, 0.5))(x)
x = Dense(10, activation='softmax', name='fc_output')(x)
model = Model(input_data, x)

model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(), metrics = ['accuracy'])
model.summary()
#model.fit(x_train, y_train, epochs=40, batch_size=128)

#model.save('./models/BNN_dropout.h5')
model = load_model('./models/BNN_dropout.h5')

pre = model.predict(x_test)

#y_test = np.argmax(y_test, axis=1)
#acc = np.mean(pre==y_test)
#print('accuracy:',acc)
#model.save('./model/BNN.h5')

#noise = np.random.rand(28,28,1)
#plt.imshow(noise,cmap='gray')
#plt.show()
def test_img(model, img):
    pre_cum = []
    pre_arg = []
    acc_cum = []
    for i in range(100):
#        print('\r','test iter:',i,end = '')
        pre = model.predict(img)
    #    pre_arg += [np.argmax(pre, axis=1)]
        pre_cum += [pre]
        pre = np.argmax(pre, axis=1)
        pre_arg += [pre]
        acc = np.mean(pre==y_test)
        acc_cum += [acc]
#    print('\n')
    pre_cum = np.array(pre_cum)
    all_prob = []
    
    flag = None
    plt.figure()
    plt.imshow(img[0,:,:,0])
    plt.xticks([])
    plt.yticks([])
    
    plt.figure(figsize=(20,2.7))
#    plt.ylim((0,1))
    for i in range(10):
    #    histo_exp = np.exp(pre_cum[:,0,i])
    #    prob = np.percentile(histo_exp, 50)
        plt.subplot(1,10,i+1)
        plt.hist(pre_cum[:,0,i])
        plt.ylim((0,100))
        plt.xlabel(i,{'size':20})
        if i == 0:
            plt.xticks([])
        else:
            plt.xticks([])
            plt.yticks([])
        
        prob = np.percentile(pre_cum[:,0,i], 50)
        print('prob ',i, ':', prob)
        all_prob.append(prob)
        if prob > 0.7:
            flag = i
    if flag != None:
        print('class:',flag)
    else:
        print('unknown')
        
    return flag
    
    
# 测试集数据测试
ind = 7
data = x_test[ind]
data = np.expand_dims(data, axis=0)
re = test_img(model, data)

    
# 随机生成数据测试
img_rand = np.random.rand(1,28,28,1)
re = test_img(model, img_rand)



# 未知数据测试
# './img/A/SWNlY3ViZS50dGY=.png'    
img = image.load_img('./img/A/SVRDIFRpZXBvbG8gQmxhY2sgSXRhbGljLnBmYg==.png', target_size=(28, 28))
img = image.img_to_array(img)
img = img/255
img = img[:,:,0]
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=3)
re = test_img(model, img)

# 测试蝶姐图片的识别效果
ind = 7
data1 = x_test[ind]
data1 = np.expand_dims(data1, axis=0)
ind = 10
data2 = x_test[ind]
data2 = np.expand_dims(data2, axis=0)
beta = 0.5
data = beta*data1 +(1-beta)*data2
re = test_img(model, data)