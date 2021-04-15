import pygame, pygame.sndarray
import numpy
import scipy.signal
from sound_api import *

############## GLOBAL VARIABLES ##############
sample_rate = 44100 # Hz
sample_packet = 4096
 
############## ALIASES FOR BITS ##############
A1 = 440    # 420-460
A2 = 494    # 
A3 = 523
A4 = 587
A5 = 659
A6 = 698
B1 = A1 * 2
B2 = A2 * 2
B3 = A3 * 2
B4 = A4 * 2
B5 = A5 * 2
B6 = A6 * 2


D1 = [5000]

# I attempt to send bits:   1111 1111, 00
# I actually receive:       1111 1111, 01

# Args: Insert a byte, play a chord, which is a function of the byte
def trans_bits8(byte):
    pass

def trans_packet100(bytearr):
    pass

def trans_file(filename):
    pass


def init():
    pygame.mixer.pre_init(sample_rate, -16, 1) # 44.1kHz, 16-bit signed, mono
    pygame.mixer.init()

if __name__ == '__main__':
    # Initialize pygame to play sounds
    init()
    
    while True:
        for fv in D1:
            play_for(sine_wave(fv, sample_packet), 1000)
        


