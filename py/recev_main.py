import scipy.fft as sfft
import numpy as np  # for arrays
import pyaudio
import time
import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt

blit = False

############## GLOBAL VARIABLES ##############
sample_rate = 48000 # Hz
sample_packet = 4096
FLU = {}    # Frequency Lookup
FFT_Threshold = 1000       # Guess based on data
NREP = 2
soft_fft_range = 6
STOP_CODON = '0000000011'

## GENERATE FREQUENCY ARRAY BASED ON INTERVAL ##
min     = 400   # Minimum frequency
itv     = 100    # Interval size
psize   = 10    # Packet size including parity bits
pamt    = 5     # Amount of packets
D = []          # Initializing of D array: int
G = []          # Initializing of G array: array<int>
for i in range(pamt): G.append([])
freqs = sfft.fftfreq(sample_packet, d=(1/sample_rate))
hexd = {
    '0000': '0',
    '0001': '1',
    '0010': '2',
    '0011': '3',
    '0100': '4',
    '0101': '5',
    '0110': '6',
    '0111': '7',
    '1000': '8',
    '1001': '9',
    '1010': 'a',
    '1011': 'b',
    '1100': 'c',
    '1101': 'd',
    '1110': 'e',
    '1111': 'f',
}


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

def enter_parseFrequencies(pyaud, stream):
    foundlt = False
    while True:
        fftres = []
        for i in range(psize): fftres.append('')

        # Read raw microphone data 
        rawsamps = stream.read(sample_packet, exception_on_overflow = False) 
        # Convert raw data to NumPy array 
        samps = np.fromstring(rawsamps, dtype=np.int16) 
        # Calculate the FFT
        pfft = sfft.rfft(samps)

        found = False
        for i in range(int(len(freqs)/2) - 1):
            # check each one that exceeds the limit
            itr = abs(np.real(pfft[i]))
            if itr > FFT_Threshold:
                found = True
                print('freq: {}\t exceeded the threshold at {}'.format(freqs[i], itr))

        if found == True and foundlt == False:
            print('parse done')
    
        fountlt = found

def readFromCRNO(cmem=None, specific_crno=0):
    # Read raw microphone data 
    rawsamps = stream.read(sample_packet, exception_on_overflow = False) 
    # Convert raw data to NumPy array 
    samps = np.fromstring(rawsamps, dtype=np.int16) 
    # Calculate the FFT
    pfft = sfft.rfft(samps)
    # Print the result of many fft results
    freqs_broken = []
    if cmem is None:
        cmem = []
        for i in range(psize): cmem.append('')

    for i in range(len(G[specific_crno])): # TODO first crno only
        # Check all outputs around the set FLU point
        thres_broken = False
        sres_arr = []
        FLU_centerpoint = FLU[G[specific_crno][i]]

        for j in range(soft_fft_range):
            # debug this
            pfft_i = FLU_centerpoint - int(soft_fft_range / 2) + j

            if abs(np.real(pfft[pfft_i])) > FFT_Threshold:  # fft output at FLU point
                #print(abs(np.real(pfft[pfft_i])), pfft_i, freqs[pfft_i])
                if str(int(freqs[pfft_i])) not in freqs_broken: freqs_broken.append(str(int(freqs[pfft_i])))
                thres_broken = True

        #print("frequency: {}\tFLU:{}\tabs(fftres): {}\tfftres: {}".format(D[i], FLU[D[i]], abs(np.real(sres)), sres))
        # conditional set to counter the the false points
        if thres_broken:
            cmem[i] = '1'
        elif cmem[i] != '1':
            cmem[i] = '0'

    return cmem

def enter_receiver(pyaud, stream, iter=0):
    print('waiting for preamble')
    enter_preambleRec(stream)

    recorded_memory = []
    lastbyte = ''

    while True:
        starttime = time.time()

        # Start of procedure
        fftres = []
        for i in range(psize): fftres.append('')

        sc_count = 0
        while True:
            fftres = readFromCRNO()

            print(fftres)
            if ''.join(fftres) == STOP_CODON:
                sc_count += 1
                if sc_count > 3:
                    write_output(recorded_memory, no=iter)
                    return

            if check_parityBits(fftres):
                break
        
        currbyte = ''.join(fftres[:len(fftres)-2])
        # if this byte is the same as last byte skip it
        print("curr: {},\tlast: {}".format(currbyte, lastbyte))
        if currbyte == lastbyte:
            continue

        recorded_memory.append(currbyte)

        lastbyte = currbyte

        elapsed = time.time() - starttime

        print('{} to {}\t=> {}\t elapsed: {},\t'.format(
            G[0][0],
            G[0][len(G[0])-1],
            ' '.join(fftres),
            elapsed,
        ))
    
# At this point, output the recorded memory to a file
def write_output(recorded_memory, no=0):
    count = 0
    filename = 'output{}.txt'.format(no)
    with open(filename, 'w') as f:
        for i in range(len(recorded_memory)):
            count += 1
            tl = translate_bin2hex(recorded_memory[i])
            f.write(tl+ ' ')
            if count == 15:
                f.write('\n')
                count = 0






def translate_bin2hex(bin):
    fnib = bin[:4]
    lnib = bin[4:]
    print(fnib, lnib, bin)

    res = hexd[fnib] + hexd[lnib]
    return res

# Receives an array, in which there are parity bits being a modulo
def check_parityBits(inp, pbits=2):
    # determine the checksum
    chksum = 0
    for i in range(len(inp) - pbits):
        if inp[i] == '1':
            chksum += 1

    # determine the parity modulo remainer
    modsum = 0
    for i in range(pbits):
        if inp[len(inp) - 1 - i] == '1':
            modsum += 2 ** i

    #print('input {}, has checksum {} and modsum {}'.format( inp, chksum, modsum))
    if chksum % 2 ** pbits == modsum:
        return True
    else:
        return False


    


def enter_preambleRec(stream):

    # Preamble code
    pac = [
        '10101010',
        '11110000',
        '00001111',
        '01010101'
    ]
    pacp = 0 # preamble code position


    while pacp < 4:
        fftres = []
        for i in range(psize): fftres.append('')
        freqs = sfft.fftfreq(sample_packet, d=(1/sample_rate))

        while True:
            fftres = readFromCRNO()

            # check parity bits, if correct no need to repeat
            if (check_parityBits(fftres)):
                #print('correct parity: {}'.format(fftres))
                break
        
        if (check_parityBits(fftres)) and ''.join(fftres) != '0000000000':
            if ''.join(fftres[:len(fftres)-2]) == pac[pacp]:
                #print('preamble #{} has been recognized: {}'.format(pacp, ''.join(fftres[:len(fftres)-2])))
                pacp += 1

        



def test_preambleRec(stream):
    min_time = 10000
    max_time = 0
    mean = 0
    over2 = 0
    over5 = 0
    over10 = 0
    ntests = 50
    for i in range(ntests):
        full_cycle_dur = 1
        starttime = time.time()
        enter_preambleRec(stream)
        elapsed = time.time() - starttime

        # Skip the first one for stats purposes
        if i == 0: continue

        if elapsed < min_time: min_time = elapsed
        if elapsed > max_time: max_time = elapsed
        if elapsed > 2: over2 += 1
        if elapsed > 5: over5 += 1
        if elapsed > 10: over10 += 1

        mean += elapsed

        print('correctly recognized preamble #{} in {} seconds'.format(i,elapsed))

    print("mean: {}, min_time: {}, max_time: {}, over2: {}, over5: {}, over10: {}, 100 tests completed".format(
        mean/ntests, min_time, max_time,over2, over5, over10))



        

def init():
    # calculate the indices for each fft component
    setup_G_arr()
    
    freqs = sfft.fftfreq(sample_packet, d=(1/sample_rate))

    for fv in D:
        index, res, diff = find_closestFFT(freqs, fv)
        #print("target: {},\tindex: {},\tres: {},\tdiff: {}".format(fv, index, res, diff))
        FLU[fv] = index

# Main runtime
if __name__ == '__main__':
    print('Starting UlSoFi-Py')

    pyaud = pyaudio.PyAudio()
    stream = pyaud.open( format = pyaudio.paInt16, channels = 1, rate = sample_rate, input_device_index = 2, input = True, frames_per_buffer=sample_packet)
    
    init()


    #enter_fftVis(pyaud, stream)
    for i in range(10):
        enter_receiver(pyaud, stream, iter=i)
    #enter_preambleRec(stream)
    #test_preambleRec(stream)

    #enter_parseFrequencies(pyaud, stream)
    

        


