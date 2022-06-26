# core packages
import streamlit as st
import streamlit.components.v1 as stc


# file processing libraries
import docx2txt
from PyPDF2 import PdfFileReader
import pdfplumber

# Load NLP packages
import spacy
from transformers import pipeline, MarianMTModel, MarianTokenizer
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
from textblob import TextBlob

# text cleaning packages
import neattext as nt
import neattext.functions as nfx


# utils
from collections import Counter
from wordcloud import WordCloud
import base64
import time
import re

timestr = time.strftime("%Y%m%d-%H%M%S")


# Function to generate sentiments

sentiment = pipeline("sentiment-analysis")


# Tweak the text cleaning function further if you wish

def clean_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\n\n", " ", text)
    text = re.sub(r"\t", " ", text)
    text = text.strip(" ")
    text = re.sub(
        " +", " ", text
    ).strip()  # get rid of multiple spaces and replace with a single
    return text  



def sentiment_analysis(my_text):
    my_text = my_text.encode("ascii", errors="ignore").decode(
        "ascii"
    )  # remove non-ascii, Chinese characters
    my_text = my_text.lower()  # lower case
    my_text = re.sub(r"\n", " ", my_text)
    my_text = re.sub(r"\n\n", " ", my_text)
    my_text = re.sub(r"\t", " ", my_text)
    my_text = my_text.strip(" ")
    my_text = re.sub(
        r"[^\w\s]", "", my_text
    )  # remove punctuation and special characters
    my_text = re.sub(
        " +", " ", my_text
    ).strip()  # get rid of multiple spaces and replace with a single
    results = sentiment(my_text)
    return results[0]["label"], round(results[0]["score"], 5)


# function for text summarization
def clean_text(text):
    text = text.encode("ascii", errors="ignore").decode(
        "ascii"
    )  # remove non-ascii and special characters
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\n\n", " ", text)
    text = re.sub(r"\t", " ", text)
    text = text.strip(" ")
    text = re.sub(" +", " ", text).strip()  # To get rid of spaces
    return text


# sumamrization function

pipeline_summ = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn",
    framework="pt",
)


def text_summarizer(text):
    input_text = clean_text(text)
    results = pipeline_summ(input_text)
    return results[0]["summary_text"]


# question answering function
reader = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


def quest_ans(question, context):
    #question = "What does the customer want?"
    outputs = reader(question, context)
    return outputs["answer"]


# function to generate named entities
def get_entities(my_text):
    docx = nlp(my_text)
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    return entities


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""


def render_entities(rawtext):
    docx = nlp(rawtext)
    html = displacy.render(docx, style="ent")
    html = html.replace("\n\n", "\n")
    result = HTML_WRAPPER.format(html)
    return result


# Function to get most common tokens
def get_most_common_tokens(my_text, num=5):
    word_tokens = Counter(my_text.split())
    most_common_tokens = dict(word_tokens.most_common(num))
    return most_common_tokens


def read_pdf(file):
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text += page.extractText()

    return all_page_text


