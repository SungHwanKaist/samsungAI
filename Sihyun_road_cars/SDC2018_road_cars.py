import tensorflow as tf
import pandas as pd
import numpy as np

#preprocessing

#read data
raw_data = pd.read_csv('dataset_noregion.csv')

X_train = np.zeros((20000, 82))
Y_train = np.zeros((20000, 15))
X_test = np.zeros((5037, 82))
Y_test = np.zeros((5037, 15))

for i in range(20000):
    X_train[i][raw_data['AM/PM'].loc[i]] += 1
    X_train[i][2 + raw_data['DAY'].loc[i]] += 1
    X_train[i][9] += raw_data['Deadpp'].loc[i]
    X_train[i][10] += raw_data['Totalpp'].loc[i]
    X_train[i][11] += raw_data['Bhurtpp'].loc[i]
    X_train[i][12] += raw_data['Shurtpp'].loc[i]
    X_train[i][13] += raw_data['Churtpp'].loc[i]
    X_train[i][14 + raw_data['Btype'].loc[i]] += 1
    X_train[i][18 + raw_data['Mtype'].loc[i]] += 1
    X_train[i][37 + raw_data['Type'].loc[i]] += 1
    X_train[i][59 + raw_data['Blaw'].loc[i]] += 1
    X_train[i][62 + raw_data['Law'].loc[i]] += 1
    #Y_train[i][raw_data['Broad'].loc[i]] += 1
    #Y_train[i][9 + raw_data['Road'].loc[i]] += 1
    #Y_train[i][raw_data['B1people'].loc[i]] += 1
    Y_train[i][raw_data['B2people'].loc[i]] += 1
##39
for i in range(20000, 25037):
    X_test[i-20000][raw_data['AM/PM'].loc[i]] += 1
    X_test[i-20000][raw_data['DAY'].loc[i]+2] += 1
    X_test[i-20000][9] += raw_data['Deadpp'].loc[i]
    X_test[i-20000][10] += raw_data['Totalpp'].loc[i]
    X_test[i-20000][11] += raw_data['Bhurtpp'].loc[i]
    X_test[i-20000][12] += raw_data['Shurtpp'].loc[i]
    X_test[i-20000][13] += raw_data['Churtpp'].loc[i]
    X_test[i-20000][14 + raw_data['Btype'].loc[i]] += 1
    X_test[i-20000][18 + raw_data['Mtype'].loc[i]] += 1
    X_test[i-20000][37 + raw_data['Type'].loc[i]] += 1
    X_test[i-20000][59 + raw_data['Blaw'].loc[i]] += 1
    X_test[i-20000][62 + raw_data['Law'].loc[i]] += 1
    #Y_test[i-20000][raw_data['Broad'].loc[i]] += 1
    #Y_test[i-20000][9 + raw_data['Road'].loc[i]] += 1
    #Y_test[i-20000][raw_data['B1people'].loc[i]] += 1
    Y_test[i-20000][raw_data['B2people'].loc[i]] += 1

    #print(preprocessed_data[:, i])

dX_train = pd.DataFrame(X_train)
dX_train.to_csv("X_train.csv", index = False)
X_train = pd.read_csv('X_train.csv', index_col = False)

dY_train = pd.DataFrame(Y_train)
dY_train.to_csv("Y_train.csv", index = False)
Y_train = pd.read_csv('Y_train.csv', index_col = False)

dX_test = pd.DataFrame(X_test)
dX_test.to_csv("X_test.csv", index = False)
X_test = pd.read_csv('X_test.csv', index_col = False)

dY_test = pd.DataFrame(Y_test)
dY_test.to_csv("Y_test.csv", index = False)
Y_test = pd.read_csv('Y_test.csv', index_col = False)


tf_x = tf.placeholder(tf.float32)
tf_y = tf.placeholder(tf.float32)

# Hidden layers
hidden_1 = 1000
hidden_2 = 1000
hidden_3 = 1000
hidden_4 = 1000
hidden_5 = 1000
hidden_6 = 1000


# Classes
n_classes = 15
# Batch size
batch_size = 64


# Model
def neural_network_model(data):
    # Weights and biases
    W1 = tf.get_variable("W1", shape=[82, hidden_1], initializer=tf.contrib.layers.xavier_initializer())
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

    W5 = tf.get_variable("W5", shape=[hidden_4, hidden_5], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([hidden_5]))
    L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)

    W6 = tf.get_variable("W6", shape=[hidden_5, hidden_6], initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.Variable(tf.random_normal([hidden_6]))
    L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)

    W7 = tf.get_variable("W7", shape=[hidden_6, n_classes], initializer=tf.contrib.layers.xavier_initializer())
    b7 = tf.Variable(tf.random_normal([n_classes]))
    output = tf.matmul(L6, W7) + b7

    return output


def train_network(x):
    # Predict
    prediction = neural_network_model(x)
    # Cross-entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y, logits=prediction)+1e-10)
    # Optimizer
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
    # Model will train for 10 cycles, feed forward + backprop
    epochs = 350

    # Session
    with tf.Session() as sess:
        # Initialize global variable
        init = tf.global_variables_initializer()
        sess.run(init)
        # Run cycles
        for epoch in range(epochs):
            epoch_loss = 0
            # Split and batch
            total_batch = int(len(X_train) / batch_size)
            X_batches = np.array_split(X_train, total_batch)
            Y_batches = np.array_split(Y_train, total_batch)
            # Run over all batches
            for i in range(total_batch):
                epoch_x, epoch_y = X_batches[i], Y_batches[i]
                _, c = sess.run([optimizer, cost], feed_dict={tf_x: epoch_x, tf_y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', epochs, 'loss', epoch_loss)

        # Check prediction
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(tf_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        result = sess.run(accuracy, feed_dict={tf_x: X_test, tf_y: Y_test})
        print("{0:f}%".format(result * 100))


train_network(tf_x)
