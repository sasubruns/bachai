import numpy as np
import tensorflow as tf
import os
import sys
import mido
import math
import datetime
import random

FEATURES = 2
ACTUAL_FEATURES = 11
BATCH_SIZE = 32

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

embedding = tf.keras.layers.Embedding(256, 10)

old_model = tf.keras.models.load_model("model.h5")

note_input = tf.keras.layers.Input(batch_shape=(1, 1, 128))
time_input = tf.keras.layers.Input(batch_shape=(1, 1, 1))

concat = tf.keras.layers.Concatenate(axis=2)([note_input, time_input])

x = tf.keras.layers.LSTM(512, return_sequences=True, stateful=True)(concat)
x = tf.keras.layers.LSTM(256, return_sequences=True, stateful=True)(x)
x = tf.keras.layers.LSTM(256, stateful=True)(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)


# Note output

n_out = tf.keras.layers.Dense(128, activation='softmax')(x)

# Time output

a_out = tf.keras.layers.Dense(1)(x)


model = tf.keras.Model([note_input, time_input], [n_out, a_out])

model.compile(loss=['categorical_crossentropy', 'mse'], loss_weights=[1, 300], optimizer=tf.keras.optimizers.Nadam(), metrics=['accuracy'])

model.set_weights(old_model.get_weights())


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def prediction(note_input, time_input, i):
    prediction = model.predict([note_input, time_input])
    note_out = prediction[0][0]
    time_out = prediction[1][0]
    return sample(note_out, 0.5), time_out


def midi_msg_from_predictions(note, time):
    ticks = max(mido.second2tick(time, 200, 50000), 0)
    msg = mido.Message('note_on', note=note, velocity=100, time=int(ticks))
    return msg

def random_one_hot(low, high, n):
    for _ in range(n):
        index = random.randint(low, high)
        one_hot = tf.one_hot(index, 128, on_value=1, off_value=0)
        yield one_hot.numpy()

def pred_to_one_hot(note):
    return tf.one_hot(note, 128, on_value=1, off_value=0).numpy()



midi_file = mido.MidiFile(type=0)
track = mido.MidiTrack()
midi_file.tracks.append(track)

note_inputs = np.vstack(random_one_hot(1, 128, 1)).reshape(1, 1, 128)
time_inputs = np.random.randint(0, 2, (1, 1, 1)) * 0.2

for i in range(400):

    note, time = prediction(note_inputs, time_inputs, i)

    msg = midi_msg_from_predictions(note, time)
    track.append(msg)
    print(".", end="")

    note_inputs, time_inputs = pred_to_one_hot(note).reshape(1, 1, 128), time.reshape(1, 1, 1)

dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
midi_file.save('outputs/output-{}.mid'.format(dt))
print(midi_file.length)