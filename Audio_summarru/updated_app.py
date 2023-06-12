# import streamlit as st
# import whisper
# from whisper.utils import get_writer
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lex_rank import LexRankSummarizer


# # Load the Whisper ASR model
# model = whisper.load_model("base")

# # Define a function to transcribe audio and video files
# def transcribe_file(file_path):
#     result = model.transcribe(file_path)
#     output_directory = "./"

#     # Save as a TXT file without any line breaks
#     with open("transcription.txt", "w", encoding="utf-8") as txt:
#         txt.write(result["text"])

#     # Save as a TXT file with hard line breaks
#     txt_writer = get_writer("txt", output_directory)
#     txt_writer(result, file_path)

#     return result["text"]

# # Define a function to summarize text
# def summarize_text(text):
#     parser = PlaintextParser.from_string(text, Tokenizer("english"))
#     summarizer = LexRankSummarizer()
#     # Summarize the document with 2 sentences
#     summary = summarizer(parser.document, 2)
#     summary_text = ""
#     for sentence in summary:
#         summary_text += str(sentence) + " "
#     return summary_text


# def main():
#     st.title("Audio and Video Transcription with Summarization")
#     st.subheader("Upload Audio or Video File")
#     uploaded_file = st.file_uploader("Choose a file", type=["mp3", "mp4"])
#     if uploaded_file is not None:
#         file_type = uploaded_file.type.split("/")[0]
#         if file_type == "audio":
#             transcription = transcribe_file(uploaded_file)
#             st.subheader("Transcription:")
#             st.write(transcription)
#         elif file_type == "video":
#             with open(uploaded_file.name, "wb") as video_file:
#                 video_file.write(uploaded_file.read())
#             transcription = transcribe_file(uploaded_file.name)
#             st.subheader("Transcription:")
#             st.write(transcription)
#         else:
#             st.warning("Invalid file type. Please upload an audio or video file.")
#         summary = summarize_text(transcription)
#         st.subheader("Summary:")
#         st.write(summary)


# if __name__ == "__main__":
#     main()
import streamlit as st
import whisper
from whisper.utils import get_writer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import os

# Load the Whisper ASR model
model = whisper.load_model("base")

# Define a function to transcribe audio and video files
def transcribe_file(file_path):
    result = model.transcribe(file_path)
    output_directory = "./"

    # Save as a TXT file without any line breaks
    with open("transcription.txt", "w", encoding="utf-8") as txt:
        txt.write(result["text"])

    # Save as a TXT file with hard line breaks
    txt_writer = get_writer("txt", output_directory)
    txt_writer(result, file_path)

    return result["text"]

# Define a function to summarize text
def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    # Summarize the document with 2 sentences
    summary = summarizer(parser.document, 2)
    summary_text = ""
    for sentence in summary:
        summary_text += str(sentence) + " "
    return summary_text


def main():
    st.title("Audio and Video Transcription with Summarization")
    st.subheader("Upload Audio or Video File")
    uploaded_file = st.file_uploader("Choose a file", type=["mp3", "mp4"])
    if uploaded_file is not None:
        file_type = uploaded_file.type.split("/")[0]
        if file_type == "audio" or file_type == "video":
            # Save the uploaded file locally
            with open(uploaded_file.name, "wb") as file:
                file.write(uploaded_file.read())
            file_path = os.path.abspath(uploaded_file.name)
            transcription = transcribe_file(file_path)
            st.subheader("Transcription:")
            st.write(transcription)
            summary = summarize_text(transcription)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Invalid file type. Please upload an audio or video file.")


if __name__ == "__main__":
    main()
