Bach AI

A recurrent neural network that learns to replicate MIDI files. Currently not very user friendly.
Data is not included. Any MIDI files that only contain one tonal instrument should work. (Multiple instruments are probably too much for the network.)
Currently does not load data in chunks (to be implemented), so limiting the number of files is recommended, unless you have massive amounts of RAM.

An audio example is included in the project. It should be noted that the program does not output audio, but MIDI.

How to use:
- change the source folder for the midi files inside open_midi.py, on line 64.
- run open_midi.py
- run train.py
- run execute.py
