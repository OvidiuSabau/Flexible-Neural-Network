import modelBackend
import imageToInput
import tensorflow as tf

#Use this to call on the function of the functions from the modelBackend.py file.
#As an example of how it works see the call below. Additionally, check the backend file for more details.
#Use the imageToInput functions to transform images into numpy arrays compatible with the model or into .csv files

final_training_cost, final_testing_cost = modelBackend.trainModelWithCSV(run_name="model2", training_file_path="./CSV/train.csv", testing_file_path="./CSV/test.csv", layer_sizes=[784, 200, 100, 50, 10], initial_learning_rate=0.001, learning_rate_decay=0.6, num_epochs=20, batch_size=64, regularization_parameter=50, save_model=True)
X_input = imageToInput.singleImageToVector("img001-001.png", save_as_csv=False)
predictions =  modelBackend.predictUsingModelWithVectors(model_path="./models/model2/model2.ckpt", layer_sizes=[784, 200, 100, 50, 10], X_input = X_input)
print(predictions)