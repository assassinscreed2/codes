1>	Imdb
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras import layers
from keras import models
from keras import optimizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
train_df = pd.read_csv("imdbdataset.csv")

# Tokenize the text data
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(train_df["review"])
x_train = tokenizer.texts_to_matrix(train_df["review"], mode="binary")

# Convert the sentiment labels to binary format
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["sentiment"]).astype("float32")

# Define the model
model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(1000,)))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

# Plot the training and validation accuracy
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plot the training and validation loss
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2>	Boston housing
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data from the CSV file
housing_df = pd.read_csv('housing.csv', header=None, delimiter='\s+')

# Split the data into features (X) and target (y)
X = housing_df.iloc[:, :-1].values
y = housing_df.iloc[:, -1].values.reshape(-1, 1)

# Scale the data using StandardScaler (scale feature and target variable)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
scaled_y = scaler.fit_transform(y)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[13]),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(0.001), metrics=['mae'])

# Train the model
history = model.fit(scaled_X, scaled_y, epochs=500, validation_split=0.2, verbose=0)


# Plot the training and validation loss over epochs
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
loss, mae = model.evaluate(scaled_X, scaled_y, verbose=0)
print("Mean absolute error:", mae)

# Predict housing prices using the model
predictions = model.predict(scaled_X)
predicted_prices = scaler.inverse_transform(predictions)

# Print predicted prices and actual prices
print("Predicted prices:", predicted_prices.flatten())
print("Actual prices:", y.flatten())
# Plot predicted prices and actual prices
plt.scatter(range(len(predicted_prices)), predicted_prices.flatten(), label='Predicted prices')
plt.scatter(range(len(y)), y.flatten(), label='Actual prices')

# Set plot title and labels
plt.title('Predicted vs Actual Prices')
plt.xlabel('Data point')
plt.ylabel('Price')

# Add legend
plt.legend()

# Show the plot
plt.show()
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
3>	Fashion-mnist
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

# Load the data from CSV files
train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

# Split the data into features and labels
x_train = train_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
x_test = test_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0

y_train = train_df["label"].values
y_test = test_df["label"].values

# Define the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define and train a deep learning model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

# Evaluate the model on the testing data
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

# Make predictions on new data
predictions = model.predict(x_test)

# Plot some images with their predictions
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                class_names[true_label]),
                                color=color)

plt.show()
plot_model(model, show_shapes=True)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Google stocks price train
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# LSTML requires data in 3d tensonr with shape (batch_size, timesteps, input_dim)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
# configure learning process
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
1>	BFS/ DFS
#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <stack>

using namespace std;

void bfs(vector<vector<int>>& graph, int start, vector<bool>& visited) {
    queue<int> q;
    q.push(start);
    visited[start] = true;

    #pragma omp parallel
    {
        #pragma omp single
        {
            while (!q.empty()) {
                int vertex = q.front();
                cout << vertex << " "<<endl;
                q.pop();

                #pragma omp task firstprivate(vertex)
                {
                    for (int neighbor : graph[vertex]) {
                        if (!visited[neighbor]) {
                            q.push(neighbor);
                            visited[neighbor] = true;
                            #pragma omp task
                            bfs(graph, neighbor, visited);
                        }
                    }
                }
            }
        }
    }
}

void parallel_bfs(vector<vector<int>>& graph, int start) {
    vector<bool> visited(graph.size(), false);
    bfs(graph, start, visited);
}

void dfs(vector<vector<int>>& graph, int start, vector<bool>& visited) {
    stack<int> s;
    s.push(start);
    visited[start] = true;
#pragma omp parallel
    {
#pragma omp single
        {
            while (!s.empty()) {
                int vertex = s.top();
                cout << vertex << " "<<endl;
                s.pop();
#pragma omp task firstprivate(vertex)
                {
                    for (int neighbor : graph[vertex]) {
                        if (!visited[neighbor]) {
                            s.push(neighbor);
                            visited[neighbor] = true;
#pragma omp task
                            dfs(graph, neighbor, visited);
                        }
                    }
                }
            }
        }
    }
}

void parallel_dfs(vector<vector<int>>& graph, int start) {
    vector<bool> visited(graph.size(), false);
    dfs(graph, start, visited);
}

int main() {
    vector<vector<int>> graph(7);
    graph[0] = {1, 2};
    graph[1] = {0, 2, 3, 4};
    graph[2] = {0, 1, 5, 6};
    graph[3] = {1, 4};
    graph[4] = {1, 3};
    graph[5] = {2};
    graph[6] = {2};

    cout << "BFS"<<endl;
    parallel_bfs(graph, 0);
    cout << "DFS"<<endl;
    parallel_dfs(graph, 0);

    return 0;
}

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <iostream>
#include <cuda_runtime.h>

__global__ void addVectors(int* A, int* B, int* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int n;
    cout << "Enter number of elements "<<endl;
    cin>>n;
    int* A, * B, * C;
    int size = n * sizeof(int);

    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    for (int i = 0; i < n; i++) {
    	cout << "Enter value in A" << endl;
        cin >> A[i];
        cout << "Enter value in B "<< endl;
        cin >> B[i];
    }

    int* dev_A, * dev_B, * dev_C;
    cudaMalloc(&dev_A, size);
    cudaMalloc(&dev_B, size);
    cudaMalloc(&dev_C, size);

    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    addVectors<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C, n);

    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 100; i++) {
        std::cout << C[i] << " - ";
    }
    std::cout << std::endl;

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

void min_reduction(vector<int>& arr) {
    int min_value = 10000;
    #pragma omp parallel for reduction(min: min_value)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] < min_value) {
            min_value = arr[i];
        }
    }
    cout << "Minimum value: " << min_value << endl;
}

void max_reduction(vector<int>& arr) {
    int max_value = -1;
    #pragma omp parallel for reduction(max: max_value)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] > max_value) {
            max_value = arr[i];
        }
    }
    cout << "Maximum value: " << max_value << endl;
}

void sum_reduction(vector<int>& arr) {
    int sum = 0;
    #pragma omp parallel for reduction(+: sum)
    for (int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    cout << "Sum: " << sum << endl;
}

void average_reduction(vector<int>& arr) {
    int sum = 0;
    #pragma omp parallel for reduction(+: sum)
    for (int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    cout << "Average: " << (double)sum / arr.size() << endl;
}

int main() {
    vector<int> arr = {5, 2, 9, 1, 7, 6, 8, 3, 4};
    min_reduction(arr);
    max_reduction(arr);
    sum_reduction(arr);
    average_reduction(arr);
}

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

void bubble_sort_odd_even(vector<int>& arr) {
    bool isSorted = false;
    while (!isSorted) {
        isSorted = true;
        #pragma omp parallel for
        for (int i = 0; i < arr.size() - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }
        #pragma omp parallel for
        for (int i = 1; i < arr.size() - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }
    }
}

void merge(vector<int>& arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
    vector<int> L(n1), R(n2);

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];

    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = j = 0;
    k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];

    while (j < n2)
        arr[k++] = R[j++];
}

void merge_sort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        merge_sort(arr, l, m);
        merge_sort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

void parallel_merge_sort(vector<int>& arr) {
#pragma omp parallel
    {
#pragma omp single
        merge_sort(arr, 0, arr.size() - 1);
    }
}

int main() {
    vector<int> arr1 = {5, 2, 9, 1, 7, 6, 8, 3, 4};
    vector<int> arr2 = {6, 1, 19, 11, 17, 61, 18, 13, 41};
    double bstart, bend, mstart, mend;
    
    bstart = omp_get_wtime();
    bubble_sort_odd_even(arr1);
    bend = omp_get_wtime();

    cout << "Parallel bubble sort using odd-even transposition time: " << bend - bstart << endl;
    
    mstart = omp_get_wtime();
    parallel_merge_sort(arr2);
    mend = omp_get_wtime();
    cout << "Parallel merge sort time: " << mend - mstart << endl;
    
    return 0;
}

