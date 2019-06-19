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
from keras.models import Model
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

x = Lambda(lambda x: K.dropout(x, 0.5))(x)

x = Dense(10, activation='softmax', name='fc_10')(x)
model = Model(input_data, x)

model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(), metrics = ['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=20, batch_size=256)

pre = model.predict(x_test)




ind = 3
data = x_test[ind]
data = np.expand_dims(data, axis=0)
pre_cum = []
pre_arg = []
acc_cum = []

for i in range(100):
    print('\r','test iter:',i,end = '')
    pre = model.predict(data)
#    pre_arg += [np.argmax(pre, axis=1)]
    pre_cum += [pre]
    pre = np.argmax(pre, axis=1)
    pre_arg += [pre]
    acc = np.mean(pre==y_test)
    acc_cum += [acc]
print('\n')

#y_test = np.argmax(y_test, axis=1)
#acc = np.mean(pre==y_test)
#print('accuracy:',acc)
#model.save('./model/BNN.h5')

#noise = np.random.rand(28,28,1)
#plt.imshow(noise,cmap='gray')
#plt.show()
pre_cum = np.array(pre_cum)
all_prob = []
#ind = 6
plt.figure(1)
plt.imshow(x_test[ind,:,:,0])
plt.figure(2)
for i in range(10):
#    histo_exp = np.exp(pre_cum[:,0,i])
#    prob = np.percentile(histo_exp, 50)
    plt.subplot(1,10,i+1)
    plt.hist(pre_cum[:,0,i])
    prob = np.percentile(pre_cum[:,0,i], 20)
    print('prob ',i, ':', prob)
#    all_prob.append(prob)
    
    

img_rand = np.random.rand(1,28,28,1)
rand_pre_cum = []
rand_pre_arg = []
radn_acc_cum = []

for i in range(100):
    print('\r','random iter:',i,end = '')
    pre = model.predict(img_rand)
#    pre_arg += [np.argmax(pre, axis=1)]
    rand_pre_cum += [pre]
#    pre = np.argmax(pre, axis=1)
#    pre_arg += [pre]
#    acc = np.mean(pre==y_test)
#    acc_cum += [acc]
print('\n')
rand_pre_cum = np.array(rand_pre_cum)
all_prob_rand = []

plt.figure(1)
plt.imshow(img_rand[0,:,:,0])
plt.figure(2)
for i in range(10):
#    histo_exp = np.exp(pre_cum[:,0,i])
#    prob = np.percentile(histo_exp, 50)
    plt.subplot(1,10,i+1)
    plt.hist(rand_pre_cum[:,0,i])
    prob = np.percentile(rand_pre_cum[:,0,i], 50)
    print('prob ',i, ':', prob)
#    all_prob.append(prob)



# './img/A/SWNlY3ViZS50dGY=.png'    
img = image.load_img('./img/A/SGVsdmV0aWNhUm91bmRlZExULUJvbGRDb25kT2JsLm90Zg==.png', target_size=(28, 28))
img = image.img_to_array(img)
img = img/255
img = img[:,:,0]
#img = cv2.transpose(img)
plt.figure()
plt.imshow(img)

pre_cum = []
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=3)
for i in range(100):
    print('\r','random iter:',i,end = '')
    pre = model.predict(img)
#    pre_arg += [np.argmax(pre, axis=1)]
    pre_cum += [pre]
    
pre_cum = np.array(pre_cum)

plt.figure()
for i in range(10):
#    histo_exp = np.exp(pre_cum[:,0,i])
#    prob = np.percentile(histo_exp, 50)
    plt.subplot(1,10,i+1)
    plt.hist(pre_cum[:,0,i])
    prob = np.percentile(pre_cum[:,0,i], 20)
    print('prob ',i, ':', prob)
    