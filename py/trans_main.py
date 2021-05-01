import pygame, pygame.sndarray
import numpy
import scipy.signal
import time
from sound_api import *

############## GLOBAL VARIABLES ##############
sample_rate = 44100 # Hz
sample_packet = 4096
 
############## ALIASES FOR BITS ##############
hexd = {
    '0': '0000',
    '1': '0001',
    '2': '0010',
    '3': '0011',
    '4': '0100',
    '5': '0101',
    '6': '0110',
    '7': '0111',
    '8': '1000',
    '9': '1001',
    'a': '1010',
    'b': '1011',
    'c': '1100',
    'd': '1101',
    'e': '1110',
    'f': '1111',
}

## GENERATE FREQUENCY ARRAY BASED ON INTERVAL ##
min     = 400   # Minimum frequency
itv     = 100    # Interval size
psize   = 10    # Packet size including parity bits
pamt    = 5     # Amount of packets
D = []          # Initializing of D array: int
G = []          # Initializing of G array: array<int>
for i in range(pamt): G.append([])

def setup_D_arr():
    for i in range(pamt * psize):
        D.append(min + i*itv)

    #Debug Printing
    print("my D: ", D)
    
    # Seperate the D arrays into G
    for i in range(len(D)):
        G[int(i / psize)].append(D[i])

    for i in range(len(G)):
        Y = []
        for x in G[i]: Y.append(str(x))
        print("Arr ",i," as follows:\t"+"\t".join(Y))


# Args: Insert a byte, play a chord, which is a function of the byte
def trans_bits8(byte, millis=1000, crno=0):
    csw = numpy.zeros(sample_rate, dtype=numpy.int16)    # Cumulative sinewave

    pcount = 0  # Count of positives [1]
    # from a given 8 bit packet, synthesize a normalized array
    for i in range(len(G[crno])):
        if i < (psize - 2): 
            if byte[i] == '1': 
                pcount += 1
                isw = sine_wave(G[crno][i], sample_packet )
                csw = numpy.add(isw, csw)

    # Normalize the Array
    # csw = numpy.true_divide(csw, pcount).astype(numpy.int16)

    play_for(csw, millis)
    pass

def trans_packet100(bytearr):
    pass

# Reads a file and transform the hexadecimal text into binary text in string
def trans_file(filename):
    with open(filename, 'r') as f:
        raw = f.read()

    # Process the raw text into pure binary
    rwonl = raw.replace('\n', ' ')     # raw without new line
    darr = rwonl.split()    # data array
    resarr = [] # result array
    for x in darr:
        n = ''  # nibble
        for c in x: # character
            n += hexd[c]
        resarr.append(n)
    
    # return the resarr
    return resarr

def init():
    pygame.mixer.pre_init(sample_rate, -16, 1) # 44.1kHz, 16-bit signed, mono
    pygame.mixer.init()

    setup_D_arr()

if __name__ == '__main__':
    # Initialize pygame to play sounds
    init()

    Barr = ['10111011', '10000000', '11001100', '11111111']

    filedata = trans_file('testfile')

    startingtime = time.time()
    for B in filedata:
        trans_bits8(B, millis=100)
    timeelapsed = time.time() - startingtime
    print("Transmission of {} bytes took {} seconds".format(len(filedata), timeelapsed))