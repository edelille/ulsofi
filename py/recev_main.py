import scipy.fft as sfft
import numpy as np  # for arrays
import pyaudio
import time
import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt

blit = False

############## GLOBAL VARIABLES ##############
sample_rate = 44100 # Hz
sample_packet = 4096
SAMPLE_AMT = 4096
FLU = {}    # Frequency Lookup
FFT_Threshold = 10000       # Guess based on data

## GENERATE FREQUENCY ARRAY BASED ON INTERVAL ##
min     = 400   # Minimum frequency
itv     = 100    # Interval size
psize   = 10    # Packet size including parity bits
pamt    = 5     # Amount of packets
D = []          # Initializing of D array: int
G = []          # Initializing of G array: array<int>
for i in range(pamt): G.append([])

def setup_G_arr():
    for i in range(pamt * psize):
        D.append(min + i*itv)

    # Seperate the D arrays into G
    for i in range(len(D)):
        G[int(i / psize)].append(D[i])

    for i in range(len(G)):
        Y = []
        for x in G[i]: Y.append(str(x))
        print("Arr ",i," as follows:\t"+"\t".join(Y))

# Visualizer meant to display
def enter_fftVis(pyaud, stream):
    plt.ion()
    X = np.linspace(0,22050,513)
    y = np.cos(X)

    fig, ax2 = plt.subplots(figsize=(8,6))
    line1, = ax2.plot(X, y)

    ax2.set_ylim([0,1000])
    text=ax2.text(0.8,0.5, "")

    plt.title("Dynamic Plot of mic input",fontsize=25)
    plt.xlabel("Frequency (Hz)",fontsize=18)
    plt.ylabel("Intensity",fontsize=18)

    fig.canvas.draw()

    # If blit, set up some blits
    if blit:
        ax2background = fig.canvas.copy_from_bbox(ax2.bbox)

    plt.show(block=False)

    t_start = time.time()
    j = 0
    while True: 
        # Read raw microphone data 
        rawsamps = stream.read(1024, exception_on_overflow = False) 
        # Convert raw data to NumPy array 
        samps = np.fromstring(rawsamps, dtype=np.int16) 
        # Calculate the FFT
        pfft = sfft.rfft(samps)

        # Parse the FFT

        ydata = [0]*int(len(pfft)/10 + 1)
        for i in range(len(pfft)):
            ydata[int(i/10)] += pfft[i]

        # FPS Text
        tx = 'Main Frame Rate:\n {fps:.3f}FPS'.format(fps= ((j+1) / (time.time() - t_start))) 
        text.set_text(tx)
        j += 1

        # DEBUG Space
        #print("Length of all: ", len(rawsamps), len(ydata), len(pfft))
        
        line1.set_xdata(X)
        line1.set_ydata(pfft)

        if blit:
            fig.canvas.restore_region(ax2background)
            ax2.draw_artist(line1)
            #ax2.draw_artist(text)
            # Fill in the axes
            fig.canvas.blit(ax2.bbox)
        else:
            fig.canvas.draw()

        fig.canvas.flush_events()

# For a given array, return the index of the closest element to a int
def find_closestFFT(arr, target):
    q = -1
    dist = 99999    # Arbitrarily large
    res = 0

    for i in range(len(arr)):
        if abs(arr[i] - target) < dist:
            dist = abs(arr[i] - target)
            res = arr[i]
            q = i
        
    return q, res, dist



def enter_receiver(pyaud, stream):
    while True:
        starttime = time.time()

        # Read raw microphone data 
        rawsamps = stream.read(SAMPLE_AMT, exception_on_overflow = False) 
        # Convert raw data to NumPy array 
        samps = np.fromstring(rawsamps, dtype=np.int16) 
        # Calculate the FFT
        pfft = sfft.rfft(samps)


        # Print the result of many fft results
        for i in range(10):
            fftres = pfft[FLU[D[i]]] # fft output at FLU point
            print("frequency: {}\tFLU:{}\tfftres: {}\tabs(fftres): {}".format(D[i], FLU[D[i]], fftres, abs(fftres)))
        


        elapsed = time.time() - starttime
        print("it took {} for one sample".format(elapsed))

        time.sleep(5)

def init():
    # calculate the indices for each fft component
    setup_G_arr()
    
    freqs = sfft.fftfreq(SAMPLE_AMT, d=(1/sample_rate))

    for fv in D:
        index, res, diff = find_closestFFT(freqs, fv)
        #print("target: {},\tindex: {},\tres: {},\tdiff: {}".format(fv, index, res, diff))
        FLU[fv] = index

# Main runtime
if __name__ == '__main__':
    print('Starting UlSoFi-Py')

    pyaud = pyaudio.PyAudio()
    stream = pyaud.open( format = pyaudio.paInt16, channels = 1, rate = 44100, input_device_index = 2, input = True, frames_per_buffer=4096)
    
    init()
    #enter_fftVis(pyaud, stream)
    enter_receiver(pyaud, stream)
    

        


