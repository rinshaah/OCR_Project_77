import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print dataset details
print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

print("Example label:", y_train[0])
print("Example image matrix:\n", x_train[0])
