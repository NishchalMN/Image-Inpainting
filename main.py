
%tensorflow_version 1.x
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# Data loading
x_train_path1 = '/content/gdrive/My Drive/Colab Notebooks/conv/cutn1.npy'
y_train_path1 = '/content/gdrive/My Drive/Colab Notebooks/conv/orig_new1.npy'

x_test_path1 = '/content/gdrive/My Drive/Colab Notebooks/conv/cutn2.npy'
y_test_path2 = '/content/gdrive/My Drive/Colab Notebooks/conv/orig_new2.npy'

x_train = np.load(x_train_path1)
y_train = np.load(y_train_path1)

x_val = np.load(x_test_path1)
y_val = np.load(y_test_path2)

# Normalizing the inputs
x_train = x_train / 255
x_val = x_val / 255

y_train = y_train /255
y_val = y_val /255


# Defining Hyperparameters
epochs = 200
learning_rate = 0.001 
batch_size = 32
n_input = 128

x = tf.placeholder("float", [None, 128,128,3])
y = tf.placeholder("float", [None, 128,128,3])


# Functions for the model
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool2d(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv2d_transpose(x, W, b, output_shape, strides=2):
    x = tf.nn.conv2d_transpose(x, W, output_shape = output_shape, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# weights and bias initialization
weights = {
    'wc1': tf.get_variable('W0', shape=(5,5,3,10), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(7,7,10,18), initializer=tf.contrib.layers.xavier_initializer()), 

    'wt1': tf.get_variable('W2', shape=(7,7,16,18), initializer=tf.contrib.layers.xavier_initializer()), 
    'wt2': tf.get_variable('W3', shape=(5,5,3,16), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(18), initializer=tf.contrib.layers.xavier_initializer()),
    
    'bc3': tf.get_variable('B2', shape=(16), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B3', shape=(3), initializer=tf.contrib.layers.xavier_initializer()),
}

def conv_net(x, weights, biases):  
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d_transpose(conv2, weights['wt1'], biases['bc3'], (32, 64, 64, 16))

    out = conv2d_transpose(conv3, weights['wt2'], biases['out'], (32, 128, 128, 3))
    
    return out

# Model Building
pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.abs(pred - y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

equality = tf.equal(pred, y)
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    for i in range(epochs):
        for batch in range(len(x_train)//batch_size):
            batch_x = x_train[batch*batch_size:min((batch+1)*batch_size,len(x_train))]
            batch_y = y_train[batch*batch_size:min((batch+1)*batch_size,len(y_train))]  
            
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        
        print("Iter " + str(i) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: x_val[:32],y : y_val[:32]})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
        
        feed_dict = {x: x_val[:32]}
        classification = sess.run(pred, feed_dict)
        # cv2.imwrite('/content/gdrive/My Drive/Colab Notebooks/TDL/test/t{}.jpg'.format(i), classification[19]*255)
    
print('Finished')