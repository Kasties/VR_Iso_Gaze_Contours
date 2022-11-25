import numpy as np
import tensorflow as tf
import math

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="isogaze.tflite")
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

# input details
print(input_details)

# output details
print(output_details)

# Test model on random input data.
input_shape = input_details[0]['shape']
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
angle = 0
magnitude = 0
percentile = 0.7
input_data = np.array([math.sin(angle), math.cos(angle), magnitude, percentile], dtype=np.float32).reshape(input_shape)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data0 = interpreter.get_tensor(output_details[0]['index'])
output_data1 = interpreter.get_tensor(output_details[1]['index'])
print(output_data0)
print(output_data1)