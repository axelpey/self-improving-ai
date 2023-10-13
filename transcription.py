import sounddevice as sd
from pydub import AudioSegment
import pygame.mixer
import time
from models import whisper


def record_audio(filename, duration, samplerate=44100):
    myrecording = sd.rec(
        int(samplerate * duration), samplerate=samplerate, channels=1, dtype="int16"
    )
    sd.wait()  # Wait until recording is finished

    # Convert recording to AudioSegment for easy export

    audio = AudioSegment(
        myrecording.tobytes(),
        frame_rate=samplerate,
        sample_width=myrecording.dtype.itemsize,
        channels=1,
    )

    audio.export(filename, format="mp3", bitrate="128k")


def play_audio(filename):
    # Initialize the mixer module
    pygame.mixer.init()
    pygame.mixer.music.load(filename)

    print(f"Playing {filename}...")
    pygame.mixer.music.play()

    # This will keep the program running while the audio plays
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    print("Playback finished.")


def record_and_transcribe(filename, duration):
    print(f"Recording for {duration} seconds...")
    record_audio(filename, duration)
    print("Recording finished. Now transcribing...")
    return transcribe_recording(filename)
