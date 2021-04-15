import scipy.fft as sfft
import numpy as np  # for arrays
# import pyaudio
# import analyse

# Main runtime
if __name__ == '__main__':
    print('Starting UlSoFi-Py')

    x = np.array(np.arange(10))
    # using scipy fft
    res = sfft.rfft(x)

    for x in res:
        print(x)

    pyaud = pyaudio.PyAudio()
    stream = pyaud.open( format = pyaudio.paInt16, channels = 1, rate = 44100, input_device_index = 2, input = True)
    
    while True: 
        # Read raw microphone data 
        rawsamps = stream.read(1024) 
        # Convert raw data to NumPy array 
        samps = numpy.fromstring(rawsamps, dtype=numpy.int16) 
        # Show the volume and pitch 
        print(analyse.loudness(samps), analyse.musical_detect_pitch(samps))

