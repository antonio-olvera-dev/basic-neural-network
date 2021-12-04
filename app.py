import tensorflow as tf
import numpy as np

# Algorithm to predict: (x*3) + 2

def trainModel(input, output, model):
    print("Training the model...")
    history = model.fit(input, output, epochs=1000, verbose=False)
    print("History loss:\n {}".format(history.history['loss']))
    print("\n")
    print("Model predict:\n {}".format(model.predict(np.array([[0], [100], [20], [30], [40]]))))


input = np.array([0, 2,  4,  8, 15, 22,  43],  dtype=float)
output = np.array([2, 8, 14, 26, 47, 68, 131], dtype=float)

layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
trainModel(input, output, model)

# print:
# Model predict:
#  [[  2.0007746]
#  [301.99796  ]
#  [ 62.00021  ]
#  [ 91.99993  ]
#  [121.99965  ]]