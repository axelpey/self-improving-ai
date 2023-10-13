from models import llm


def query_llm(text):
    llm_input = (
        f"[INST]You are my AI partner Jarvis. Here is my prompt for you: {text}[/INST]"
    )

    # print(llm_input)
    output = llm(llm_input)

    return output["choices"][0]["text"]


def is_instruction(text):
    llm_input = (
        "[INST]You are an AI that's designed to recognize when a user is calling you for assistance. If they don't ask you, it's NOT a request or instruction. "
        "Determine if the following transcript contains a direct instruction or request for your help:\n\n"
        f"'{text}'"
        "\n\nReply with 'True' if there's a direct instruction or request, and 'False' if not. Please be exact in your response.[/INST]"
    )

    # print(llm_input)
    output = llm(llm_input)

    return (
        True
        if "True" in output["choices"][0]["text"]
        else (False if "False" in output["choices"][0]["text"] else None)
    )


def is_jarvis_instruction(text):
    llm_input = (
        "[INST]You are Jarvis, an AI that's designed to recognize when a user is calling you for assistance. A call for assistance must include an explicit call to your name, 'Jarvis' "
        "Determine if the following transcript contains a direct instruction or request for your help:\n\n"
        f"'{text}'"
        "\n\nReply with 'True' if there's a direct instruction or request, and 'False' if not. Please be exact in your response.[/INST]"
    )

    # print(llm_input)
    output = llm(llm_input)

    return (
        True
        if "True" in output["choices"][0]["text"]
        else (False if "False" in output["choices"][0]["text"] else None)
    )
