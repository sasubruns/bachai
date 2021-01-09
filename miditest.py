import os
import sys
import mido

midi_file = mido.MidiFile(type=0)
track = mido.MidiTrack()
midi_file.tracks.append(track)

msg1 = mido.Message('note_on', note=67, velocity=100, time=10000)
msg2 = mido.Message('note_on', note=67, velocity=100, time=10000)

track.append(msg1)
track.append(msg2)

midi_file.save('test.mid')

