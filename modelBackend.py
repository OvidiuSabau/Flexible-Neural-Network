from CSV2Array import readInputsAndOutputs, readInputs
import tensorflow as tf
import numpy as np

#The variable layer_sizes refers to a list of ints, corresponding to the number of layers (including the input layer) and the size of each layer
#For example [784, 50, 10] means that the input has 784 , the first hidden layer has 50 nodes and the output has 10 nodes (in this case a softmax)

def trainModelWithCSV(run_name, layer_sizes, training_file_path, testing_file_path, initial_learning_rate, learning_rate_decay, num_epochs, batch_size, regularization_parameter, save_model = False):

    # Code to reset the tensorflow graph & make tensorflow release VRAM after it's done computing
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #Importing the data from the specified paths. Can be changed to adjust for what you want to compute
    X_training, Y_training = readInputsAndOutputs(training_file_path)
    X_testing, Y_testing  = readInputsAndOutputs(testing_file_path)

    #Calculating the number of batches based on the batchsize specified
    m = X_training.shape[0]
    if m//batch_size == m/batch_size:
        num_batches = m//batch_size
    else:
        num_batches = m//batch_size + 1

    #Dictionaries that are used to save intermediate values of the graph
    a = dict()
    z = dict()
    weights = dict()
    biases = dict()
    a_normalized = dict()

    #Defines the model according to the layer sizes specified
    for index, layer_size in enumerate(layer_sizes):
        if index == 0:
            #Creates placeholder for the X's. Adds that value to a dictionary for easier computing afterwards
            with tf.variable_scope('input'):
                X = tf.placeholder(dtype=tf.float32, shape=[None, layer_size], name='X')
                a_normalized['a_normalized0'] = X
        else:
            #Defines computations for each layer in the model
            with tf.variable_scope('layer'+str(index)):

                #Initializes weights using Xavier Initialization
                weights['w'+str(index)] = tf.get_variable(name='weights'+str(index), dtype=tf.float32, shape=[layer_sizes[index-1], layer_sizes[index]], initializer=tf.contrib.layers.xavier_initializer())

                #Initializes biases to 0
                biases['b'+str(index)] = tf.get_variable(name='biases'+str(index), dtype=tf.float32, shape=[layer_sizes[index]], initializer = tf.zeros_initializer())

                #Computes the linear activation
                z['z' + str(index)] = tf.matmul(a_normalized['a_normalized'+str(index-1)], weights['w'+str(index)]) + biases['b'+str(index)]

                #Computes the non-linear activation for all layers except for the last one
                if index != len(layer_sizes) - 1:
                    a['a' + str(index)] = tf.nn.relu(z['z'+str(index)])
                    a_normalized['a_normalized'+str(index)] = tf.layers.batch_normalization(inputs=a['a'+str(index)], axis=1)

                # Activation of the last layer. Can be changed according what you want to predict
                else:
                    outputs = tf.nn.softmax(logits= z['z'+str(index)])

    #Computes the sum of frobenius norm of all the weights matrixes
    weights_squarred_sum = 0
    for index in range(1,len(layer_sizes)):
        weights_squarred_sum += tf.norm(weights["w"+str(index)], ord='fro', axis=[-2, -1])

    #Defines the cost function. Change according to last layer's activation. Additional calculations for regularization
    with tf.variable_scope('cost'):
            Y = tf.placeholder(dtype=tf.float32, shape=(None, layer_sizes[len(layer_sizes)-1]), name='Y')
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z['z'+str(len(layer_sizes)-1)], labels=Y) + regularization_parameter/(2*m)*weights_squarred_sum)

    #Defines optimizer
    with tf.variable_scope('optimizer'):
            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #Object used to log cost's across runs and epochs. Used later
    with tf.variable_scope('logging'):
        tf.summary.scalar(name='cost', tensor=cost)
        summary = tf.summary.merge_all()

    #If specified, saves model. Used later
    if save_model:
        saver = tf.train.Saver()

    #Starts a session
    with tf.Session(config=config) as session:

        #Initializes all the variables (weights and biases)
        session.run(tf.global_variables_initializer())

        #Objects used to write log files for training and testing costs
        training_writer = tf.summary.FileWriter("./logs/" + run_name + "/training", session.graph)
        testing_writer  = tf.summary.FileWriter("./logs/" + run_name + "/testing", session.graph)

        #Training loop running according to the specified number of epochs
        for epoch in range(num_epochs):
            for batch in range(num_batches):

                #Selecting batch to run optimizer on
                X_training_batch = X_training[batch*batch_size:(batch+1)*batch_size, :]
                Y_training_batch = Y_training[batch*batch_size:(batch+1)*batch_size, :]

                #Runs one step of the Adam optimizer for every batch
                session.run([optimizer], feed_dict={X: X_training_batch, Y: Y_training_batch, learning_rate: initial_learning_rate / (1 + learning_rate_decay * epoch)})

            #Logs training and testing costs every 5 epochs
            if epoch % 5 == 0:
                training_cost, training_summary = session.run([cost, summary], feed_dict={X: X_training, Y: Y_training})
                testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_testing, Y: Y_testing})
                print("Epoch #" + str(epoch) + ": training cost= " + str(training_cost) + " testing cost= " + str(testing_cost))
                training_writer.add_summary(training_summary, epoch)
                testing_writer.add_summary(testing_summary, epoch)

        #Display percentage of accurate predictions
        predictions = session.run(outputs,feed_dict={X : X_testing})
        expected = np.argmax(Y_testing, axis=1)
        predictions = np.argmax(predictions, axis=1)
        correct = 0
        for index in range(len(predictions)):
            if predictions[index] == expected[index]:
                correct += 1
        print("Testing accuracy = " + str(correct/(len(predictions))*100) + "%")

        #If specified, saves model
        if save_model:
            saver.save(sess=session, save_path="./models/" + run_name +"/" + run_name + ".ckpt")
            f = open("./models/" + run_name + "/" + "layer_sizes.txt", "w+")
            f.write(str(layer_sizes))
            f.close()

        return session.run(cost, feed_dict={X: X_training, Y: Y_training}), session.run(cost, feed_dict={X: X_testing, Y: Y_testing})

def traingModelWithVectors(run_name, layer_sizes, X_training, Y_training, X_testing, Y_testing, initial_learning_rate, learning_rate_decay, num_epochs, batch_size, regularization_parameter, save_model = False):

    # Code to reset the tensorflow graph & make tensorflow release VRAM after it's done computing
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #Calculating the number of batches based on the batchsize specified
    m = X_training.shape[0]
    if m//batch_size == m/batch_size:
        num_batches = m//batch_size
    else:
        num_batches = m//batch_size + 1

    #Dictionaries that are used to save intermediate values of the graph
    a = dict()
    z = dict()
    weights = dict()
    biases = dict()
    a_normalized = dict()

    #Defines the model according to the layer sizes specified
    for index, layer_size in enumerate(layer_sizes):
        if index == 0:
            #Creates placeholder for the X's. Adds that value to a dictionary for easier computing afterwards
            with tf.variable_scope('input'):
                X = tf.placeholder(dtype=tf.float32, shape=[None, layer_size], name='X')
                a_normalized['a_normalized0'] = X
        else:
            #Defines computations for each layer in the model
            with tf.variable_scope('layer'+str(index)):

                #Initializes weights using Xavier Initialization
                weights['w'+str(index)] = tf.get_variable(name='weights'+str(index), dtype=tf.float32, shape=[layer_sizes[index-1], layer_sizes[index]], initializer=tf.contrib.layers.xavier_initializer())

                #Initializes biases to 0
                biases['b'+str(index)] = tf.get_variable(name='biases'+str(index), dtype=tf.float32, shape=[layer_sizes[index]], initializer = tf.zeros_initializer())

                #Computes the linear activation
                z['z' + str(index)] = tf.matmul(a_normalized['a_normalized'+str(index-1)], weights['w'+str(index)]) + biases['b'+str(index)]

                #Computes the non-linear activation for all layers except for the last one
                if index != len(layer_sizes) - 1:
                    a['a' + str(index)] = tf.nn.relu(z['z'+str(index)])
                    a_normalized['a_normalized'+str(index)] = tf.layers.batch_normalization(inputs=a['a'+str(index)], axis=1)

                # Activation of the last layer. Can be changed according what you want to predict
                else:
                    outputs = tf.nn.softmax(logits= z['z'+str(index)])

    #Computes the sum of frobenius norm of all the weights matrixes
    weights_squarred_sum = 0
    for index in range(1,len(layer_sizes)):
        weights_squarred_sum += tf.norm(weights["w"+str(index)], ord='fro', axis=[-2, -1])

    #Defines the cost function. Change according to last layer's activation. Additional calculations for regularization
    with tf.variable_scope('cost'):
            Y = tf.placeholder(dtype=tf.float32, shape=(None, layer_sizes[len(layer_sizes)-1]), name='Y')
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z['z'+str(len(layer_sizes)-1)], labels=Y) + regularization_parameter/(2*m)*weights_squarred_sum)

    #Defines optimizer
    with tf.variable_scope('optimizer'):
            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #Object used to log cost's across runs and epochs. Used later
    with tf.variable_scope('logging'):
        tf.summary.scalar(name='cost', tensor=cost)
        summary = tf.summary.merge_all()

    #If specified, saves model. Used later
    if save_model:
        saver = tf.train.Saver()

    #Starts a session
    with tf.Session(config=config) as session:

        #Initializes all the variables (weights and biases)
        session.run(tf.global_variables_initializer())

        #Objects used to write log files for training and testing costs
        training_writer = tf.summary.FileWriter("./logs/" + run_name + "/training", session.graph)
        testing_writer  = tf.summary.FileWriter("./logs/" + run_name + "/testing", session.graph)

        #Training loop running according to the specified number of epochs
        for epoch in range(num_epochs):
            for batch in range(num_batches):

                #Selecting batch to run optimizer on
                X_training_batch = X_training[batch*batch_size:(batch+1)*batch_size, :]
                Y_training_batch = Y_training[batch*batch_size:(batch+1)*batch_size, :]

                #Runs one step of the Adam optimizer for every batch
                session.run([optimizer], feed_dict={X: X_training_batch, Y: Y_training_batch, learning_rate: initial_learning_rate / (1 + learning_rate_decay * epoch)})

            #Logs training and testing costs every 5 epochs
            if epoch % 5 == 0:
                training_cost, training_summary = session.run([cost, summary], feed_dict={X: X_training, Y: Y_training})
                testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_testing, Y: Y_testing})
                print("Epoch #" + str(epoch) + ": training cost= " + str(training_cost) + " testing cost= " + str(testing_cost))
                training_writer.add_summary(training_summary, epoch)
                testing_writer.add_summary(testing_summary, epoch)

        #Display percentage of accurate predictions
        predictions = session.run(outputs,feed_dict={X : X_testing})
        expected = np.argmax(Y_testing, axis=1)
        predictions = np.argmax(predictions, axis=1)
        correct = 0
        for index in range(len(predictions)):
            if predictions[index] == expected[index]:
                correct += 1
        print("Testing accuracy = " + str(correct/(len(predictions))*100) + "%")

        #If specified, saves model
        if save_model:
            saver.save(sess=session, save_path="./models/" + run_name +"/" + run_name + ".ckpt")
            f = open("./models/" + run_name + "/" + "layer_sizes.txt", "w+")
            f.write(str(layer_sizes))
            f.close()

        return session.run(cost, feed_dict={X: X_training, Y: Y_training}), session.run(cost, feed_dict={X: X_testing, Y: Y_testing})


def predictUsingModelWithCSV(model_path, layer_sizes, input_file_path):

    # Code to reset the tensorflow graph & make tensorflow release VRAM after it's done computing
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Importing the data from the specified paths. Can be changed to adjust for what you want to compute
    X_input = readInputs(input_file_path)

    # Dictionaries that are used to save intermediate values of the graph
    a = dict()
    z = dict()
    weights = dict()
    biases = dict()
    a_normalized = dict()

    # Defines the model according to the layer sizes specified
    for index, layer_size in enumerate(layer_sizes):
        if index == 0:
            # Creates placeholder for the X's. Adds that value to a dictionary for easier computing afterwards
            with tf.variable_scope('input'):
                X = tf.placeholder(dtype=tf.float32, shape=[None, layer_size], name='X')
                a_normalized['a_normalized0'] = X
        else:
            # Defines computations for each layer in the model
            with tf.variable_scope('layer' + str(index)):

                # Initializes weights using Xavier Initialization
                weights['w' + str(index)] = tf.get_variable(name='weights' + str(index), dtype=tf.float32, shape=[layer_sizes[index - 1], layer_sizes[index]], initializer=tf.contrib.layers.xavier_initializer())

                # Initializes biases to 0
                biases['b' + str(index)] = tf.get_variable(name='biases' + str(index), dtype=tf.float32, shape=[layer_sizes[index]], initializer=tf.zeros_initializer())

                # Computes the linear activation
                z['z' + str(index)] = tf.matmul(a_normalized['a_normalized' + str(index - 1)], weights['w' + str(index)]) + biases['b' + str(index)]

                # Computes the non-linear activation for all layers except for the last one
                if index != len(layer_sizes) - 1:
                    a['a' + str(index)] = tf.nn.relu(z['z' + str(index)])
                    a_normalized['a_normalized' + str(index)] = tf.layers.batch_normalization(inputs=a['a' + str(index)], axis=1)

                # Activation of the last layer. Can be changed according what you want to predict
                else:
                    outputs = tf.nn.softmax(logits=z['z' + str(index)])

    # Computes the sum of frobenius norm of all the weights matrixes
    weights_squarred_sum = 0
    for index in range(1, len(layer_sizes)):
        weights_squarred_sum += tf.norm(weights["w" + str(index)], ord='fro', axis=[-2, -1])

    saver = tf.train.Saver()

    # Starts a session
    with tf.Session(config=config) as session:

        # Initializes all the variables (weights and biases)
        saver.restore(sess=session, save_path=model_path)

        #Compute predicitions for the inputs
        predicitons = session.run(outputs,feed_dict={X: X_input})
        return np.argmax(predicitons, axis=1)

def predictUsingModelWithVectors(model_path, layer_sizes, X_input):

    # Code to reset the tensorflow graph & make tensorflow release VRAM after it's done computing
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Dictionaries that are used to save intermediate values of the graph
    a = dict()
    z = dict()
    weights = dict()
    biases = dict()
    a_normalized = dict()

    # Defines the model according to the layer sizes specified
    for index, layer_size in enumerate(layer_sizes):
        if index == 0:
            # Creates placeholder for the X's. Adds that value to a dictionary for easier computing afterwards
            with tf.variable_scope('input'):
                X = tf.placeholder(dtype=tf.float32, shape=[None, layer_size], name='X')
                a_normalized['a_normalized0'] = X
        else:
            # Defines computations for each layer in the model
            with tf.variable_scope('layer' + str(index)):

                # Initializes weights using Xavier Initialization
                weights['w' + str(index)] = tf.get_variable(name='weights' + str(index), dtype=tf.float32, shape=[layer_sizes[index - 1], layer_sizes[index]], initializer=tf.contrib.layers.xavier_initializer())

                # Initializes biases to 0
                biases['b' + str(index)] = tf.get_variable(name='biases' + str(index), dtype=tf.float32, shape=[layer_sizes[index]], initializer=tf.zeros_initializer())

                # Computes the linear activation
                z['z' + str(index)] = tf.matmul(a_normalized['a_normalized' + str(index - 1)], weights['w' + str(index)]) + biases['b' + str(index)]

                # Computes the non-linear activation for all layers except for the last one
                if index != len(layer_sizes) - 1:
                    a['a' + str(index)] = tf.nn.relu(z['z' + str(index)])
                    a_normalized['a_normalized' + str(index)] = tf.layers.batch_normalization(inputs=a['a' + str(index)], axis=1)

                # Activation of the last layer. Can be changed according what you want to predict
                else:
                    outputs = tf.nn.softmax(logits=z['z' + str(index)])

    # Computes the sum of frobenius norm of all the weights matrixes
    weights_squarred_sum = 0
    for index in range(1, len(layer_sizes)):
        weights_squarred_sum += tf.norm(weights["w" + str(index)], ord='fro', axis=[-2, -1])

    saver = tf.train.Saver()

    # Starts a session
    with tf.Session(config=config) as session:

        # Initializes all the variables (weights and biases)
        saver.restore(sess=session, save_path=model_path)

        #Compute predicitions for the inputs
        predicitons = session.run(outputs,feed_dict={X: X_input})
        return np.argmax(predicitons, axis=1)