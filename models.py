# %%

import os, ujson
from faster_whisper import WhisperModel
from llama_cpp import Llama
from openai import Audio, ChatCompletion
import openai


openai.api_key = ujson.load(open("creds.json"))["openai_api_key"]


def init_mistral_whispermed_models():
    llm = Llama(
        model_path="models/mistral-7b-instruct-v0.1.gguf", n_threads=6, verbose=False
    )

    def transcribe_recording(filename):
        segments = whisper.transcribe(filename)
        transcribed_segments = list(segments)
        transcribed_text = " ".join([segment.text for segment in transcribed_segments])
        return transcribed_text

    # Add the CT2_VERBOSE=1 flag to the environment variables
    os.environ["CT2_VERBOSE"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    model_size = "medium.en"
    whisper = WhisperModel(model_size, device="cpu", compute_type="int8")

    def whisper_func(filename):
        return transcribe_recording(filename)

    def llm_func(input):
        formatted_input = f"[INST]{input}[/INST]"
        return llm(formatted_input)["choices"][0]["text"]

    return llm_func, whisper_func


def init_mistral_whispertiny_models():
    llm = Llama(
        model_path="models/mistral-7b-instruct-v0.1.gguf", n_threads=6, verbose=False
    )

    def transcribe_recording(filename):
        segments = whisper.transcribe(filename)
        transcribed_segments = list(segments)
        transcribed_text = " ".join([segment.text for segment in transcribed_segments])
        return transcribed_text

    # Add the CT2_VERBOSE=1 flag to the environment variables
    os.environ["CT2_VERBOSE"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    model_size = "base.en"
    whisper = WhisperModel(model_size, device="cpu", compute_type="int8")

    def whisper_func(filename):
        return transcribe_recording(filename)

    def llm_func(input):
        formatted_input = f"[INST]{input}[/INST]"
        return llm(formatted_input)["choices"][0]["text"]

    return llm_func, whisper_func


def init_gpt_35_whispertiny_models():
    def transcribe_recording(filename):
        segments = whisper.transcribe(filename)
        transcribed_segments = list(segments)
        transcribed_text = " ".join([segment.text for segment in transcribed_segments])
        return transcribed_text

    # Add the CT2_VERBOSE=1 flag to the environment variables
    os.environ["CT2_VERBOSE"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    model_size = "base.en"
    whisper = WhisperModel(model_size, device="cpu", compute_type="int8")

    def whisper_func(filename):
        return transcribe_recording(filename)

    def gpt35(input):
        completion = ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": input}]
        )
        return completion.choices[0].message.content

    return gpt35, whisper_func


def init_web_openai_models():
    class AdaptedWhisper:
        def __init__(self):
            pass

        def transcribe(self, filename):
            return Audio.transcribe("whisper-1", open(filename, "rb")).text

    def gpt35(input):
        completion = ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": input}]
        )
        return completion.choices[0].message.content

    return gpt35, AdaptedWhisper()


# llm, whisper = init_mistral_whispermed_models()

llm, whisper = init_web_openai_models()

# %%
