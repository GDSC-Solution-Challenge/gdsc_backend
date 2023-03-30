import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="baseOX_230318.tflite")
interpreter.allocate_tensors()

# Print input shape and type
print(interpreter.get_input_details()[0]['shape'])  # Example: [1 224 224 3]
print(interpreter.get_input_details()[0]['dtype'])  # Example: <class 'numpy.float32'>

# Print output shape and type
print(interpreter.get_output_details()[0]['shape'])  # Example: [1 3]
print(interpreter.get_output_details()[0]['dtype'])  # Example: <class 'numpy.float32'>