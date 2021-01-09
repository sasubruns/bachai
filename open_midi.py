import numpy as np
import tensorflow as tf
import os
import sys
import mido
import math
import random
from pathlib import Path

INPUT_LENGTH = 35

N_FILES = 100

def yield_midi(files_list, directory):

    files_length = len(files_list)

    buffer = []
    catch_counter = 0

    for i, file in enumerate(files_list):

        filename = os.fsdecode(file)
        print('Reading {}'.format(filename))

        #Midi from file
        try:
            mid = mido.MidiFile(os.path.join(os.fsdecode(directory), filename))

            counter = 0

            #Iterate through messages in midi
            for msg in mid:

                if (msg.type == 'note_on'):

                    time = msg.time + counter
                    counter = 0
                    if (time > 0):
                        buffer = []
                    embeddable = msg.note
                    if (embeddable in buffer):
                        continue
                    buffer.append(embeddable)
                    yield np.hstack([time, tf.one_hot(msg.note, 128, on_value=1, off_value=0).numpy()])

                elif (msg.type == 'note_off'):

                    counter = counter + msg.time

        except:
            print('Skipped file #{} due to exception'.format(i))
            catch_counter = catch_counter + 1
            print('Skips so far: {}'.format(catch_counter))
            continue


 
        print('{}/{} done'.format(i, files_length))

#Data from files
def load_song_data():

    path = "D:/bach/mid"
    directory = os.fsencode(path)

    #List containing all .mid files in /bach/mid
    all_files = list(Path(path).rglob("*.[mM][iI][dD]"))
    files_list = random.sample(all_files, min(len(all_files), N_FILES))
    

    #Iterate through mid files
    data = np.stack(yield_midi(files_list, directory), axis=0)

    print('Done loading files')
    return data


def data_to_io(data):

    note_inputs = list()
    time_inputs = list()
    note_outputs = list()
    time_outputs = list()

    length = len(data)

    # Create a sliding window of the data for the model
    for i in range(length - INPUT_LENGTH):

        # 0 to INPUT_LENGTH index of window as inputs, INPUT_LENGTH index as output
        window = data[i:i+INPUT_LENGTH + 1]
        note_inp = np.asarray(window[:-1])[:,range(1, 129)]
        time_inp = np.asarray(window[:-1])[:,0]
        note_inputs.append(note_inp)
        time_inputs.append(time_inp)
        note_outputs.append(np.asarray(window[-1][range(1, 129)]))
        time_outputs.append(np.asarray(window[-1][0]))



        if (i % 100000 == 0):

            print("{}/{} processed".format(i, length))

    print("Done with processing")
    
    # Return the arrays such that their length is divisible by 32, which is the batch size (ignore some values)
    return np.asarray(time_inputs, dtype=np.float32)[:(len(time_inputs)//32)*32], np.asarray(note_inputs, dtype=np.float32)[:(len(time_inputs)//32)*32], np.asarray(note_outputs, dtype=np.float32)[:(len(note_outputs)//32)*32], np.asarray(time_outputs, dtype=np.float32)[:(len(time_outputs)//32)*32]

midi_data = load_song_data()
print(midi_data.shape)

time_inputs, note_inputs, note_outputs, time_outputs = data_to_io(midi_data)

print(time_inputs.shape)
print(note_inputs.shape)
print(time_outputs.shape)
print(note_outputs.shape)

np.save("time_inputs.npy", time_inputs)
np.save("note_inputs.npy", note_inputs)
np.save("note_outputs.npy", note_outputs)
np.save("time_outputs.npy", time_outputs)