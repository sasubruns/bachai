Bach AI

A recurrent neural network using Tensorflow, NumPy and Mido that learns to replicate MIDI files.

Data is not included. Any MIDI files that only contain one tonal instrument should work great. (Multiple instruments are probably too much for the network.)

Currently does not load data in chunks (to be implemented), so limiting the number of files is recommended, unless you have massive amounts of RAM.

An audio example is included in the project. It should be noted that the program does not output audio, but MIDI.

Usage (CLI is also yet to be implemented, sadly):
- change the source folder for the midi files inside open_midi.py, on line 64.
- run open_midi.py
- run train.py
- run execute.py
