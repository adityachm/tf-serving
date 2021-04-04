# importing libraries
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Loading dataset
(train_images,train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Use 10000 images for training and 1000 images for testing
train_labels = train_labels[:10000]
test_labels = test_labels[:1000]

train_images = train_images[:10000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# define the model
model = Sequential()
model.add(Dense(512,activation='relu',input_shape = (784,)))
model.add(Dense(10,activation='softmax'))

# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

# Train the model
model.fit(train_images, train_labels, epochs=50, validation_data=(test_images,test_labels))

# Evaluate Model
loss,acc = model.evaluate(test_images, test_labels, verbose=2)
print("model, accuracy: {:5.2f}%".format(100*acc))

# Save the model
model.save('models/1')