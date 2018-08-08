import tensorflow as tf
import pandas as pd
import numpy as np

#preprocessing

#read data
raw_data = pd.read_csv('dataset_noregion.csv')

X_train = np.zeros((20000, 105))
Y_train = np.zeros((20000, 4))
X_test = np.zeros((5037, 105))
Y_test = np.zeros((5037, 4))

for i in range(20000):
    X_train[i][raw_data['AM/PM'].loc[i]] += 1
    X_train[i][2 + raw_data['DAY'].loc[i]] += 1
    X_train[i][9 + raw_data['Btype'].loc[i]] += 1
    X_train[i][13 + raw_data['Mtype'].loc[i]] += 1
    X_train[i][32 + raw_data['Type'].loc[i]] += 1
    X_train[i][54 + raw_data['Blaw'].loc[i]] += 1
    X_train[i][57 + raw_data['Law'].loc[i]] += 1
    X_train[i][77 + raw_data['B1people'].loc[i]] += 1
    X_train[i][90 + raw_data['B2people'].loc[i]] += 1

    Y_train[i][0] += raw_data['Deadpp'].loc[i]
    Y_train[i][1] += raw_data['Bhurtpp'].loc[i]
    Y_train[i][2] += raw_data['Shurtpp'].loc[i]
    Y_train[i][3] += raw_data['Churtpp'].loc[i]

##39
for i in range(20000, 25037):
    X_test[i-20000][raw_data['AM/PM'].loc[i]] += 1
    X_test[i-20000][2 + raw_data['DAY'].loc[i]] += 1
    X_test[i-20000][9 + raw_data['Btype'].loc[i]] += 1
    X_test[i-20000][13 + raw_data['Mtype'].loc[i]] += 1
    X_test[i-20000][32 + raw_data['Type'].loc[i]] += 1
    X_test[i-20000][54 + raw_data['Blaw'].loc[i]] += 1
    X_test[i-20000][57 + raw_data['Law'].loc[i]] += 1
    X_test[i-20000][77 + raw_data['B1people'].loc[i]] += 1
    X_test[i-20000][90 + raw_data['B2people'].loc[i]] += 1

    Y_test[i-20000][0] += raw_data['Deadpp'].loc[i]
    Y_test[i-20000][1] += raw_data['Bhurtpp'].loc[i]
    Y_test[i-20000][2] += raw_data['Shurtpp'].loc[i]
    Y_test[i-20000][3] += raw_data['Churtpp'].loc[i]

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
n_classes = 4
# Batch size
batch_size = 64


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
    cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf_y, predictions=prediction)+1e-10)
    # Optimizer
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
    # Model will train for 10 cycles, feed forward + backprop
    epochs = 5

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
        subtract = tf.subtract(prediction, tf_y)
        result = sess.run(subtract, feed_dict={tf_x: X_test, tf_y: Y_test})
        print(result)


train_network(tf_x)