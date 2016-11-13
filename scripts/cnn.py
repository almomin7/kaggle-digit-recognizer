import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
from sklearn.preprocessing import OneHotEncoder

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def define_model(restore_model_path):
    sess = tf.InteractiveSession()
    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # Weight/bias variables
    x_image = tf.reshape(x, [-1,28,28,1])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    # Defines the convolution layers
    # conv,max_pool,conv,max_pool,full layer,dropout,final layer
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    saver = tf.train.Saver()

    # Run the model
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    predictor = tf.argmax(y_conv, 1)

    # Restore from previous session
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, restore_model_path)
    # Return the training step, and the accuracy and prediction calcs
    return sess, x, y_, keep_prob, train_step, accuracy, predictor

def train_batch(traindf, x, y_, keep_prob, i, train_step, accuracy):
    one_hot_encoder = OneHotEncoder(n_values=10,dtype=np.float32,sparse=False)
    trainX = traindf.as_matrix(columns=incols)
    trainX = trainX / 255.0
    trainY = traindf.as_matrix(columns=outcols)
    one_hot_encoder.fit(trainY)
    trainY = one_hot_encoder.transform(trainY)
    batchX = trainX
    batchY = trainY
    if i%10 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x:batchX, y_: batchY, keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 0.5})

def save_model(sess, model_path):
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_path)
    print ("Model saved in file %s" % save_path)

def cross_validate_model(num_train, restore_model_path):
    df = pd.read_csv('../data/train.csv')
    testdf = df[num_train:]

    testX = testdf.as_matrix(columns=incols)
    testX = testX / 255.0
    testY = testdf.as_matrix(columns=outcols)
    one_hot_encoder = OneHotEncoder(n_values=10,dtype=np.float32,sparse=False)
    one_hot_encoder.fit(testY)
    oneHotTestY = one_hot_encoder.transform(testY)
    
    sess, x, y_, keep_prob, train_step, \
        accuracy, predictor = define_model(restore_model_path)
    
    print("Result of test: Accuracy = %g"%accuracy.eval(feed_dict={
        x: testX, y_: oneHotTestY, keep_prob: 1.0}))

    predictions = predictor.eval(feed_dict={x: testX, keep_prob:1.0}, session=sess)

    pred_out = pd.DataFrame()
    pred_out["ImageId"] = range(num_train,42000)
    pred_out["Label"] = testY
    pred_out["Prediction"] = predictions
    print pred_out.head(10)

def train_model(N, M, num_train, dosave, restore_model_path, save_model_path):

    df = pd.read_csv('../data/train.csv')
    tf.logging.set_verbosity(tf.logging.INFO)

    sess, x, y_, keep_prob, train_step, \
        accuracy, predictor = define_model(restore_model_path)

    print "%d iterations of size %d on data of length %d" % (N, M, num_train)
    for i in range(N):
        traindf = df[0:num_train].sample(n=M,replace=False)
        train_batch(traindf, x, y_, keep_prob, i, train_step, accuracy)

    if dosave: save_model(sess, save_model_path)

def predict_model(restore_model_path, start, N):
    testdf = pd.read_csv('../data/test.csv')
    testX = testdf.as_matrix(columns=incols)[start-1:start+N-1]
    testX = testX / 255.0

    print "Restore model..."
    sess, x, y_, keep_prob, train_step, \
        accuracy, predictor = define_model(restore_model_path)
    print ("Model restored from %s" % restore_model_path)
    
    print "Predicting..."
    predictions = predictor.eval(feed_dict={x: testX, keep_prob:1.0})
    print "Predictions complete."

    pred_out = pd.DataFrame()
    pred_out["ImageId"] = range(start,start + len(predictions))
    pred_out["Label"] = predictions
    print pred_out.head(10)

    out_file = ("../data/cnn_%05d-%05d.csv" % (start, start+N-1))
    pred_out.to_csv(out_file, index=False)
    print "Results written to %s" % out_file

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Train, xvalidate, process a CNN')
    parser.add_argument('operation',type=str)
    parser.add_argument('--start',type=int,default=1)
    parser.add_argument('--N',type=int,default=10000)
    parser.add_argument('--batches',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=1000)
    args = parser.parse_args()

    op = args.operation

    incols=[]
    for i in range(784):
        incols.append("pixel"+str(i))
    outcols=["label"]

    if op == 'train':
        batch_size = args.batch_size
        batches = args.batches
        num_train=35000
        train_model(batches, batch_size, num_train, True, "./models/cnn/1/model.ckpt", \
            "./models/cnn/1/model.ckpt") 

    if op == 'validate':
        cross_validate_model(num_train, "./models/cnn/1/model.ckpt") 
    
    if op == 'completetraining':
        batch_size = args.batch_size
        batches = args.batches
        num_train=42000
        train_model(batches, batch_size, num_train, True, "./models/cnn/1/model.ckpt", \
            "./models/cnn/1complete/model.ckpt") 

    if op == 'predict':
        start = args.start
        num = args.N
        predict_model("./models/cnn/1complete/model.ckpt", start, num)
