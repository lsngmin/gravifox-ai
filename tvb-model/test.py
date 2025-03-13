import tensorflow as tf
from data_loader import get_testData_generators
loaded_model = tf.keras.models.load_model('xception_model-2.h5')

test = get_testData_generators()

predictions = loaded_model.predict(test)
loss, accuracy = loaded_model.evaluate(test)

print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

print(predictions)