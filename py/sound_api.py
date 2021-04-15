import pygame, pygame.sndarray
import numpy
import scipy.signal

############## GLOBAL VARIABLES ##############
sample_rate = 44100 # Hz
sample_packet = 4096


""" Play the given NumPy array, as a sound, for ms milliseconds. """
def play_for(sample_wave, ms):
    sound = pygame.sndarray.make_sound(sample_wave.astype(int))
    sound.play(-1)
    pygame.time.delay(ms)
    sound.stop()

"""
Compute N samples of a sine wave with given frequency and peak amplitude.
Defaults to one second.
"""
def sine_wave(hz, peak, n_samples=sample_rate):
    length = sample_rate / float(hz)
    omega = numpy.pi * 2 / length
    xvalues = numpy.arange(int(length)) * omega
    onecycle = peak * numpy.sin(xvalues)
    return numpy.resize(onecycle, (n_samples,)).astype(numpy.int16)

"""
Make a chord based on a list of frequency ratios.
using a given waveform (defaults to a sine wave).
"""
def make_chord(hz, ratios, waveform=None):
    if not waveform:
        waveform = sine_wave
    chord = waveform(hz, sample_packet)
    for r in ratios[1:]:
        chord = sum([chord, waveform(hz * r / ratios[0], sample_packet)])
    return chord
