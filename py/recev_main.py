import scipy.fft as sfft
import numpy as np  # for arrays
import pyaudio
import time
import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt

blit = True


# Main runtime
if __name__ == '__main__':
    print('Starting UlSoFi-Py')

    pyaud = pyaudio.PyAudio()
    stream = pyaud.open( format = pyaudio.paInt16, channels = 1, rate = 44100, input_device_index = 2, input = True, frames_per_buffer=4096)
    

    plt.ion()
    X = np.linspace(0,22050,205)
    y = np.cos(X)

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    img = ax1.imshow(X, vmin=-1, vmax=1, interpolation="None", cmap="RdBu")

    line1 = ax2.plot([], lw=3)

    ax1.set_xlim(0, 22000)
    ax2.set_ylim([0,1000])

    plt.title("Dynamic Plot of mic input",fontsize=25)
    plt.xlabel("Frequency (Hz)",fontsize=18)
    plt.ylabel("Intensity",fontsize=18)

    fig.canvas.draw()

    # If blit, set up some blits
    if blit:
        axbackground = fig.canvas.copy_from_bbox(ax1.bbox)
        ax2background = fig.canvas.copy_from_bbox(ax2.bbox)

    plt.show(block=False)

    while True: 
        # Read raw microphone data 
        rawsamps = stream.read(4096, exception_on_overflow = False) 
        # Convert raw data to NumPy array 
        samps = np.fromstring(rawsamps, dtype=np.int16) 
        # Calculate the FFT
        pfft = sfft.rfft(samps)

        # Parse the FFT
        ydata = [0]*int(len(pfft)/10 + 1)
        for i in range(len(pfft)):
            ydata[int(i/10)] += pfft[i]


        # DEBUG Space
        #print("Length of all: ", len(rawsamps), len(ydata), len(pfft))
        
        
        line1.set_xdata(X)
        line1.set_ydata(ydata)
        

        if blit:
            fig.canvas.restore_regeion(axbackground)
            fig.canvas.restore_regeion(ax2background)

            ax1.draw_artist(img)
            ax2.draw_artist(line)
            ax2.draw_artist(text)

            # Fill in the axes
            fig.canvas.blit(ax1.bbox)
            fig.canvas.blit(ax2.bbox)

        else:
            fig.canvas.draw()

        fig.canvas.flush_events()

        


