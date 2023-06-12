import streamlit as st
import whisper
from whisper.utils import get_writer
from tempfile import NamedTemporaryFile
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Load the Whisper ASR model
model = whisper.load_model("base")

# Define a function to transcribe audio and save as TXT
def transcribe_audio(audio_file):
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_file.flush()
        result = model.transcribe(temp_file.name)
        transcription = result["text"]

        # Save as a TXT file without any line breaks
        with open("transcription.txt", "w", encoding="utf-8") as txt:
            txt.write(transcription)

        # Save as a TXT file with hard line breaks
        txt_writer = get_writer("txt", "./")
        txt_writer(result, temp_file.name)

    return transcription

# Define a function to generate summary of text file
def summary_file(file_path):
    parser = PlaintextParser.from_file(file_path, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    # Summarize the document with 2 sentences
    summary = summarizer(parser.document, 2)
    summary_text = ''
    for sentence in summary:
        summary_text += str(sentence) + '\n'
    return summary_text

# Create a Streamlit app
def main():
    st.title("Audio summary")

    # Add file upload option
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if uploaded_file is not None:
        # Transcribe audio and display result
        transcription = transcribe_audio(uploaded_file)
        st.write("Transcription:")
        st.write(transcription)

        # Generate summary of text file
        summary_text = summary_file("transcription.txt")
        st.write("Summary:")
        st.write(summary_text)

if __name__ == "__main__":
    main()
