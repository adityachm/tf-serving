import json
import requests
from tensorflow.keras import datasets

(train_images,train_labels), (test_images, test_labels) = datasets.mnist.load_data()

test_labels = test_labels[:1000]

test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# create a json string to ask query to the depoyed model
data = json.dumps({"signature_name": "serving_default",
                   "instances": test_images[0:3].tolist()})


# headers for the post request
headers = {"content-type": "application/json"}

# make the post request 
json_response = requests.post('http://localhost:9000/v1/models/mnist/versions/1:predict',
                              data=data,
                              headers=headers)

# get the predictions
predictions = json.loads(json_response.text)
print(predictions)