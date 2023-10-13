import threading
import time
import os
from pathlib import Path
import re
from typing import List
from llm_queries import is_jarvis_instruction, query_llm
from models import whisper

from transcription import record_audio


def retrieve_last_n_seconds(
    data: str, duration_per_chunk: int, total_seconds: int
) -> List[str]:
    """Retrieves the transcriptions of the last n seconds (or equivalent
    duration) from given data."""
    # Assuming each chunk is of uniform duration
    chunks_needed = total_seconds // duration_per_chunk

    # Use regular expressions to find all unprocessed chunks
    pattern = re.compile(
        r"<chunk n=(\d+) processed=(\d+)>\n(.*?)\n<\/chunk>", re.DOTALL
    )
    matches = re.findall(pattern, data)

    last_chunks = matches[-chunks_needed:]

    return last_chunks


class AudioProcessor:
    def __init__(self, duration=30):
        self.duration = duration
        self.chunk_id = 0
        self.run_id = time.strftime("%Y-%m-%d_%H:%M:%S")
        self.data_folder = Path(f"data/run_{self.run_id}")
        self.txt_file = str(self.data_folder / f"transcriptions_{self.run_id}.txt")
        self.is_transcription_locked = False
        self.running = True

    def stop(self):
        self.running = False

    def record_audio_chunked(self):
        while self.running:
            f_name = str(self.data_folder / f"chunk_{self.chunk_id}.mp3")
            print(f"Recording new chunk: {self.chunk_id}")
            record_audio(f_name, self.duration)
            self.chunk_id += 1
            # time.sleep(self.duration)
        print("Recording stopped.")

    def transcribe_audio(self):
        processed_chunks = 0
        while self.running:
            if processed_chunks < self.chunk_id:
                f_name = str(self.data_folder / f"chunk_{processed_chunks}.mp3")
                transcription = whisper.transcribe(f_name)

                # Sometimes, the transcription is hallucinating and contains
                # "Thank you for watching" or some alternative.
                if "Thank" in transcription and "watching" in transcription:
                    transcription = "[HALLUCINATION]"

                while (
                    self.is_transcription_locked
                ):  # Wait until ready to write in transcription file
                    time.sleep(0.1)

                with open(self.txt_file, "a") as f:
                    f.write(
                        f"<chunk n={processed_chunks} processed=0>\n{transcription}\n</chunk>\n"
                    )
                processed_chunks += 1
            time.sleep(1)
        print("Transcription stopped.")

    def process_transcriptions(self):
        text_to_study = ""

        while self.running:
            if not Path(self.txt_file).exists():
                time.sleep(1)
                continue

            with open(self.txt_file, "r") as f:
                data = f.read()

                # Now the goal is to first understand if we have a request or not.
                # To do this, we can use the is_instruction function.
                last_30_seconds_data = retrieve_last_n_seconds(data, self.duration, 30)
                print("Last 30 seconds of data available:", last_30_seconds_data)

                # Now pick in the last chunks those that are not hallucinations
                last_valid_chunks = [
                    chunk
                    for chunk in last_30_seconds_data
                    if chunk[2] != "[HALLUCINATION]"
                ]

                # Now pick in the last chunks those have not been processed yet
                last_unprocessed_chunks = [
                    chunk for chunk in last_valid_chunks if chunk[1] == "0"
                ]

                # Finally, get the text of the last unprocessed chunks
                new_text_to_study = " ".join(
                    [chunk[2] for chunk in last_unprocessed_chunks]
                )

                if new_text_to_study == text_to_study:
                    time.sleep(0.5)
                    continue

                text_to_study = new_text_to_study

                print("New text to study:", text_to_study)

                if is_jarvis_instruction(text_to_study):
                    # First mark the chunks as processed
                    # Lock the file, read it, replace the chunks, write it back
                    self.is_transcription_locked = True

                    with open(self.txt_file, "r") as f:
                        data = f.read()

                    for chunk in last_unprocessed_chunks:
                        data = data.replace(chunk[0], f"{chunk[0]} processed=1")

                    with open(self.txt_file, "w") as f:
                        f.write(data)

                    self.is_transcription_locked = False

                    # Provide the answer accordingly
                    response = query_llm(text_to_study)
                    response = response.replace("'", '"')
                    os.system(f"say '{response}'")

            time.sleep(1)
        print("Processing stopped.")

    def run(self):
        self.data_folder.mkdir(parents=True, exist_ok=True)

        try:
            threading.Thread(target=self.transcribe_audio).start()
            threading.Thread(target=self.process_transcriptions).start()
            self.record_audio_chunked()

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt. Stopping threads...")
            self.stop()


processor = AudioProcessor(duration=10)
processor.run()
