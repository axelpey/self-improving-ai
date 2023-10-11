# self-improving-ai
An AI running on my computer that learns how to help me

## Setup

### Environment
Either you install the official PyPi `llama-cpp-python` with `pip install llama-cpp-python` or you build it yourself. To build it yourself, you can just do the following:

```bash
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
make build
```

Then install the requirements from `requirements.txt` with `pip install -r requirements.txt`.

### LLM

I use Mistral-7B-v0.1, more specifically the weights quantized by the great TheBloke in [https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF](in his HF repo for Mistral-7B-v0.1) in version Q5-K-M