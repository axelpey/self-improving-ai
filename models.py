import os
from faster_whisper import WhisperModel
from llama_cpp import Llama


def init_models():
    llm = Llama(
        model_path="models/mistral-7b-instruct-v0.1.gguf", n_threads=6, verbose=False
    )

    # Add the CT2_VERBOSE=1 flag to the environment variables
    os.environ["CT2_VERBOSE"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    model_size = "medium.en"
    whisper = WhisperModel(model_size, device="cpu", compute_type="int8")

    return llm, whisper


llm, whisper = init_models()
