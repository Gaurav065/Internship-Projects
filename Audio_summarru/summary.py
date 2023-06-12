import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


def summary_file(File):
    parser = PlaintextParser.from_file(File, Tokenizer("english"))
    summarizer = LexRankSummarizer()
#Summarize the document with 2 sentences
    summary = summarizer(parser.document, 2)
    for sentence in summary:
        print(sentence)
summary_file("transcription.txt")
