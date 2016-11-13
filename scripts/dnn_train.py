import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.learn as tflearn

tf.logging.set_verbosity(tf.logging.INFO)

incols=[]
for i in range(784):
    incols.append("pixel"+str(i))
    outcols=["label"]

traindf = pd.read_csv('../data/train.csv')

# Let's normalize inputs between 0 and 1
#traindf /= (1 + traindf)

trainX = traindf.as_matrix(columns=incols)
trainX = trainX / float(1 + trainX.max())
trainY = traindf.as_matrix(columns=outcols)

print traindf[["label"]].head()

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=784)]
# Logging setup
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    trainX,
    trainY,
    every_n_steps=1)

num_hidden_neurons = 20
max_num_steps = 15000
model_dir = "models/3/"
sv_chkpts_secs = 30

dnn = tflearn.DNNClassifier(feature_columns=feature_columns,
                            hidden_units=[num_hidden_neurons],
                            n_classes=10,
                            model_dir=model_dir,
                            enable_centered_bias=True,
                            config=tf.contrib.learn.RunConfig(
                                save_checkpoints_secs=sv_chkpts_secs))
print "Fitting..."
dnn.fit(x=trainX, y=trainY,max_steps=max_num_steps, monitors=[validation_monitor])
print "Fitted..."

testdf = pd.read_csv('../data/test.csv')
testX = testdf.as_matrix(columns=incols)
#testY = testdf.as_matrix(columns=outcols)

pred = dnn.predict(testX[0:5])

print pred
