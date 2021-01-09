import numpy as np
import tensorflow as tf


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


INPUT_LENGTH = 50
BATCH_SIZE = 32

note_inputs = np.load("note_inputs.npy")
time_inputs = np.load("time_inputs.npy")
note_outputs = np.load("note_outputs.npy")
time_outputs = np.load("time_outputs.npy")

print(note_inputs.shape)
print(time_inputs.shape)
print(note_outputs.shape)
print(time_outputs.shape)

# Model



note_input = tf.keras.layers.Input((INPUT_LENGTH, 128))
time_input = tf.keras.layers.Input((INPUT_LENGTH, 1))

concat = tf.keras.layers.Concatenate(axis=2)([note_input, time_input])

x = tf.keras.layers.LSTM(512, return_sequences=True)(concat)
x = tf.keras.layers.LSTM(256, return_sequences=True)(x)
x = tf.keras.layers.LSTM(256)(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)


# Note output

n_out = tf.keras.layers.Dense(128, activation='softmax')(x)

# Time output

a_out = tf.keras.layers.Dense(1)(x)


model = tf.keras.Model([note_input, time_input], [n_out, a_out])

model.compile(loss=['categorical_crossentropy', 'mse'], loss_weights=[1, 300], optimizer=tf.keras.optimizers.Nadam(), metrics=['accuracy'])

model.fit([note_inputs, time_inputs], [note_outputs, time_outputs], epochs=40)
model.save('model.h5')
model.summary()
