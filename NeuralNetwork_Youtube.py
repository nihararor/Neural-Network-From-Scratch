import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#creating of neural Network
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        #seting input, hidden and output layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #randomly selecting input hidden and hidden output weights
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))


        self.hidden_layer_output = None

        self.losses = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

#implementing of forword bias here
    def forward(self, inputs):
      
        hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(hidden_layer_input)

       
        output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        final_output = self.sigmoid(output_layer_input)

        return final_output
#implementing of backward Propogation
    def backward(self, inputs, targets, output, learning_rate):
        
        error = targets - output

        output_delta = error * self.sigmoid_derivative(output)
        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)

        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_layer_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate
#use to run the epochs and to find the loss of data
    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            #it triggering forward
            output = self.forward(inputs)

           #it is basically triggering or calling the backward propogation
            self.backward(inputs, targets, output, learning_rate)

            #here we are mean squaring the error and appending the loss
            loss = np.mean(0.5 * (targets - output) ** 2)
            self.losses.append(loss)

            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, inputs):
        return self.forward(inputs)
#here we are finding the mean square error, mean absolute error and root mea square error.
    def evaluate(self, predictions, targets):
        mse = np.mean((targets - predictions) ** 2)
        mae = np.mean(np.abs(targets - predictions))
        rmse = np.sqrt(mse)
        return mse, mae, rmse
#accuracy finding task is implementing here in this we are taking data in the form of binary data
    def accuracy(self, predictions, targets, threshold=0.5):
        binary_predictions = (predictions > threshold).astype(int)
        accuracy = np.mean(binary_predictions == targets.reshape(-1, 1))
        return accuracy

# Reading CSV file for the data Prediction
data = pd.read_csv('India_Youtube.csv')

#Preprocessing Process ib which data is in the form of binary digit means likes are is binary if its 1 then trending or if 0 then not trending.
threshold_likes = 10000
data['trending'] = (data['likes'] > threshold_likes).astype(int)

#so features we used for data store is views_count, comment_count, categoryID and taking target value as trending.
#mainly it is a Pandas function so, that it can particularly work on selected dataset.
features = data[['view_count', 'comment_count', 'categoryId']]
target = data['trending']

#It is sk learn part which basically used for mean and standar deviation in data where mean is 0 and standar deviation is 1.
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)  #transform is for computation mean and standard deviation in dataonly

#here data is splitting in train and test
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42) 

#self defined parameter used in NN
input_size = X_train.shape[1]
hidden_size = 8
output_size = 1
learning_rate = 0.01

#Calling NeuralNetwork
nn = NeuralNetwork(input_size, hidden_size, output_size)

#calling training model
nn.train(X_train, y_train.values.reshape(-1, 1), epochs=500, learning_rate=learning_rate)

#calling Predictions testing
predictions = nn.predict(X_test)

#calling evaluation function like mean square error, mean absolute error and root mean square error and Accuracy part
mse, mae, rmse = nn.evaluate(predictions, y_test.values.reshape(-1, 1))
accuracy = nn.accuracy(predictions, y_test.values.reshape(-1, 1))

#Output we are showing
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
