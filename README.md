Bach AI

A recurrent neural network that learns to replicate MIDI files. Currently not very user friendly.
Data is not included. Any MIDI files that only contain one tonal instrument should work. (Multiple instruments are probably too much for the network.)
Data folder can be set inside open_midi.py, on line 64. Currently does not load data in chunks (to be implemented), so limiting the number of files is recommended, unless you have massive amounts of RAM.
