import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.learn as tflearn

num_hidden_neurons_1 = 30
num_hidden_neurons_2 = 20
max_num_steps = 4500
model_dir = "models/small/1/"
sv_chkpts_secs = 5
num_train=37000
num_cross_check=42000-num_train
num_subsets=37000

prob = num_subsets / (42000.0 - num_cross_check)

tf.logging.set_verbosity(tf.logging.INFO)

incols=[]
for i in range(784):
    incols.append("pixel"+str(i))
    outcols=["label"]

df = pd.read_csv('../data/train.csv')
traindf = df[0:num_train].sample(frac=prob,replace=False)
#testdf = df[num_train:].sample(frac=prob,replace=False)
testdf = df[num_train:]

# Let's normalize inputs between 0 and 1
#traindf /= (1 + traindf)
trainX = traindf.as_matrix(columns=incols)
trainX = trainX / 255.0
trainY = traindf.as_matrix(columns=outcols)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=784)]
# Logging setup
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    trainX,
    trainY,
    every_n_steps=1)


dnn = tflearn.DNNClassifier(feature_columns=feature_columns,
                            hidden_units=[num_hidden_neurons_1],
                            #hidden_units=[num_hidden_neurons_1,num_hidden_neurons_2],
                            n_classes=10,
                            model_dir=model_dir,
                            enable_centered_bias=True,
                            config=tf.contrib.learn.RunConfig(
                                save_checkpoints_secs=sv_chkpts_secs))
print "Fitting..."
dnn.fit(x=trainX, y=trainY,max_steps=max_num_steps, monitors=[validation_monitor])
print "Fitted..."

testX = testdf.as_matrix(columns=incols)
testX = testX / 255.0
testY = testdf.as_matrix(columns=outcols)
pred = dnn.predict(testX)

print len(pred), len(testdf)

testresultsdf = pd.DataFrame()
testresultsdf["Prediction"] = pred
testresultsdf["Label"] = testY
testresultsdf["Match"] = testresultsdf["Prediction"] == testresultsdf["Label"]

print testresultsdf.head(1000)
print "Accuracy? ", len(testresultsdf[testresultsdf["Match"]]) / (1.0 * len(testresultsdf))
print "Match vs total ", len(testresultsdf[testresultsdf["Match"]]), len(testresultsdf)
