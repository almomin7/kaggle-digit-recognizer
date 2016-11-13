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

testdf = pd.read_csv('../data/test.csv')

testX = testdf.as_matrix(columns=incols)
testX = testX / float(1 + testX.max())

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=784)]

num_hidden_neurons = 30
model_dir = "models/small/1"

dnn = tflearn.DNNClassifier(feature_columns=feature_columns,
                            hidden_units=[num_hidden_neurons],
                            n_classes=10,
                            model_dir=model_dir,
                            enable_centered_bias=True,
                            )
# Load directly from file - do we need the other params?

print "Prediction..."
pred = dnn.predict(x=testX)
print "Predicted."
print pred
pred_out = pd.DataFrame()
pred_out["ImageId"] = range(1,28001)
pred_out["Label"] = pred
pred_out.to_csv('../data/dnn_30hidden.csv', index=False)
