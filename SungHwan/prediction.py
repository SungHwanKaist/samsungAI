import tensorflow as tf
import pandas as pd
import numpy as np

#read data
raw_data = pd.read_csv('test.csv')

X_test = np.zeros((11, 105))

for i in range(11):
    X_test[i][raw_data['AM/PM'].loc[i]] += 1
    X_test[i][2 + raw_data['DAY'].loc[i]] += 1
    X_test[i][9 + raw_data['Btype'].loc[i]] += 1
    X_test[i][13 + raw_data['Mtype'].loc[i]] += 1
    X_test[i][32 + raw_data['Type'].loc[i]] += 1
    X_test[i][54 + raw_data['Blaw'].loc[i]] += 1
    X_test[i][57 + raw_data['Law'].loc[i]] += 1
    X_test[i][77 + raw_data['B1people'].loc[i]] += 1
    X_test[i][90 + raw_data['B2people'].loc[i]] += 1


dX_test = pd.DataFrame(X_test)
dX_test.to_csv("X_test.csv", index = False)
X_test = pd.read_csv('X_test.csv', index_col = False)


tf_x = tf.placeholder(tf.float32)

hidden_1 = 100
hidden_2 = 100
hidden_3 = 100
hidden_4 = 100


# Classes
n_classes = 4


# Model
def neural_network_model(data):
    # Weights and biases
    W1 = tf.get_variable("W1", shape=[105, hidden_1], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([hidden_1]))
    L1 = tf.nn.relu(tf.matmul(data, W1) + b1)

    W2 = tf.get_variable("W2", shape=[hidden_1, hidden_2], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([hidden_2]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    W3 = tf.get_variable("W3", shape=[hidden_2, hidden_3], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([hidden_3]))
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

    W4 = tf.get_variable("W4", shape=[hidden_3, hidden_4], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([hidden_4]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

    W5 = tf.get_variable("W5", shape=[hidden_4, n_classes], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([n_classes]))
    output = tf.nn.relu(tf.matmul(L4, W5) + b5)


    return output

def predict(x):
    # Predict
    prediction = neural_network_model(x)

    # Session
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.import_meta_graph('./checkpoint/checkpoint_450.meta')
        saver.restore(sess, './checkpoint/checkpoint_450')

        result = sess.run(prediction, feed_dict={tf_x: X_test})

        print(result)


predict(tf_x)