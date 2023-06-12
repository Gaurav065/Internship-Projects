import whisper
from whisper.utils import get_writer


model = whisper.load_model("base")
audio = "./Audio_summarru/test.mp3"
result = model.transcribe(audio)
output_directory = "./"


# Save as a TXT file without any line breaks
with open("transcription.txt", "w", encoding="utf-8") as txt:
    txt.write(result["text"])


# Save as a TXT file with hard line breaks
txt_writer = get_writer("txt", output_directory)
txt_writer(result, audio)
