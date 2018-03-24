import tensorflow as tf
import numpy as np

#Number of classes(15 ziektes)
n_classes = 15

#Batches die hij selecteert en langs gaat
batch_size = 32

x = tf.placeholder('float')#, [None, 65536])#65536
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

#Convolutional step
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Pool step with the conv
def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#CNN model definition
def convolutional_neural_network(x):
    #make weight dict with two hidden layers, een fc(full layer) layer en een output.
    weights = {'W_conv1': tf.Variable(tf.random_normal([4, 4, 1, 16])),
               'W_conv2': tf.Variable(tf.random_normal([4, 4, 16, 32])),
               'W_fc': tf.Variable(tf.random_normal([32 * 32 * 32, 512])),
               'out': tf.Variable(tf.random_normal([512, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([16])),
              'b_conv2': tf.Variable(tf.random_normal([32])),
              'b_fc': tf.Variable(tf.random_normal([512])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    #reshape image to 256*256
    x = tf.reshape(x, shape=[-1, 256, 256, 1])

    #perform conv step
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    #pool conv1
    conv1 = maxpool2d(conv1)

    # perform conv step
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    #pool conv2
    conv2 = maxpool2d(conv2)

    #for fc layer
    fc = tf.reshape(conv2, [-1, 32 * 32 * 32])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    #perform dropout
    fc = tf.nn.dropout(fc, keep_rate)
    #
    #make output layer
    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

#train sequence with input x = data
def train_neural_network(xData, yData):
    print(len(xData))
    #Convert X_train to float32
    X_train = tf.cast(xData, tf.float32)
    y_train = tf.cast(yData, tf.float32)
    # print(" test "))
    #prediction with model
    prediction = convolutional_neural_network(X_train)

    #calculate cost
    new_y = tf.cast(y, tf.int32)
    entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=new_y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(entropy)

    hm_epochs = 1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(xData):
                start = i
                end = i + batch_size
                batch_x = np.array(xData[start:end])
                batch_y = np.array(yData[start:end])
                _, c = sess.run([optimizer, entropy], feed_dict={x: batch_x,y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:', accuracy.eval({x: x, y: y}))


    return

def test_neural_network(x):
    return