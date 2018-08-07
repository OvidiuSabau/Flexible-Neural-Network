import pandas as pd
import tensorflow as tf

#Used to read both inputs and outputs (labels)
def readInputsAndOutputs(path):
    data = pd.read_csv(path)
    Y = data['label'].values
    X = data.drop(labels='label', axis=1).values
    Y = tf.one_hot(Y, depth=10)
    with tf.Session() as session:
        return X, session.run(Y)

#Used to only read inputs
def readInputs(path):
    data = pd.read_csv(path)
    X = data.values
    return X