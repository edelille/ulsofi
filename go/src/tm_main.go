package main

import (
    "fmt"
    "time"
    "strings"

    // Transmitter
    //"github.com/padster/go-sound/sounds"
	//"github.com/padster/go-sound/output"

    // Receiver
    "github.com/gordonklaus/portaudio"
)
func custom_check(e error) {
    if e != nil {
        panic(e)
    }
}


func init() {
    // Initialize portaudio
    portaudio.Initialize()
}





func main() {
    fmt.Println("Helloworld")

    /* debugs for the transmitter
    xSound := sounds.SumSounds(aSound, bSound, cSound)

    fmt.Println("Playing A440")
    output.Play(xSound)
    */

    info, err := portaudio.DefaultOutputDevice()
    custom_check(err)

    println(info.Name, info.MaxInputChannels, info.MaxOutputChannels, info.DefaultSampleRate)

    buffer := make([]int32, 64)
    stream, err := portaudio.OpenDefaultStream(
        1,
        0,
        44100,
        len(buffer),
        &buffer,
    )
    custom_check(err)


    // Start the stream
    custom_check(stream.Start())

    for {
        custom_check(stream.Read())

        println(strings.Join(strings.Fields(fmt.Sprint(buffer[:32]))," "))
        time.Sleep(1 * time.Second)
    }


    // Eventually close the stream
    defer stream.Close()


    // Endtime
    portaudio.Terminate()
}
