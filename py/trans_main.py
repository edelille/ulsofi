import pygame, pygame.sndarray
import numpy
import scipy.signal
from sound_api import *

############## GLOBAL VARIABLES ##############
sample_rate = 44100 # Hz
sample_packet = 4096


def init():
    pygame.mixer.pre_init(sample_rate, -16, 1) # 44.1kHz, 16-bit signed, mono
    pygame.init()

if __name__ == '__main__':
    # Initialize pygame to play sounds
    init()

    
    play_for(sine_wave(440, sample_packet), 1000)
