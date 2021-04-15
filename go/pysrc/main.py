import pyaudio
import numpy as np

CHUNK = 4096
SR = 44100

if __name__ == '__main__':
    p = pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paInt16,
        channels=1,
        rate=SR,
        input=True,
        frames_per_buffer=CHUNK
    ) #uses default input device
    
    print("hello")
