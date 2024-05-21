import numpy as np
from sentence_transformers import SentenceTransformer, util
# %%
import pandas as pd

df = pd.read_csv('data.csv')

question = df['Question']
answer = df['Answer']
# %%
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# %%
embeddings = model.encode(question)
# %%
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# %%
tokenizerId = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-id-en")
modelId = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-id-en")

# %%
tokenizerEnId = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-id")
modelEnId = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-id")
# %%
tokenizerInc = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-inc-en")
modelInc = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-inc-en")
# %%
tokenizerEnInc = AutoTokenizer.from_pretrained("thilina/mt5-sinhalese-english")
modelEnInc = AutoModelForSeq2SeqLM.from_pretrained("thilina/mt5-sinhalese-english")
# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizerVi = AutoTokenizer.from_pretrained("VietAI/envit5-translation")
modelVi = AutoModelForSeq2SeqLM.from_pretrained("VietAI/envit5-translation")
# %%
input_ids = tokenizerEnInc("How are you", return_tensors="pt").input_ids
outputs = modelEnInc.generate(input_ids=input_ids)
translatedSentence = tokenizerEnInc.batch_decode(outputs, skip_special_tokens=True)


# %%
def translateToEng(sentence, originLanguage):
    if (originLanguage == 'English'):
        return sentence
    elif (originLanguage == 'Vietnamese'):
        input_ids = tokenizerVi(sentence, return_tensors="pt", padding=True).input_ids
        outputs = modelVi.generate(input_ids=input_ids, max_length=512)
        translatedSentence = tokenizerVi.batch_decode(outputs, skip_special_tokens=True)
        return translatedSentence[0][4:]
    elif (originLanguage == 'Indonesian'):
        input_ids = tokenizerId(sentence, return_tensors="pt", padding=True).input_ids
        outputs = modelId.generate(input_ids=input_ids, max_length=512)
        translatedSentence = tokenizerId.batch_decode(outputs, skip_special_tokens=True)
        return translatedSentence[0]
    elif (originLanguage == 'Sinhala'):
        input_ids = tokenizerInc(sentence, return_tensors="pt", padding=True).input_ids
        outputs = modelInc.generate(input_ids=input_ids, max_length=512)
        translatedSentence = tokenizerInc.batch_decode(outputs, skip_special_tokens=True)
        return translatedSentence[0]


# %%
def translateFromEng(sentence, originLanguage):
    if (originLanguage == 'English'):
        return sentence
    elif (originLanguage == 'Vietnamese'):
        input_ids = tokenizerVi(sentence, return_tensors="pt", padding=True).input_ids
        outputs = modelVi.generate(input_ids=input_ids, max_length=512)
        translatedSentence = tokenizerVi.batch_decode(outputs, skip_special_tokens=True)
        return translatedSentence[0][4:]
    elif (originLanguage == 'Indonesian'):
        input_ids = tokenizerEnId(sentence, return_tensors="pt", padding=True).input_ids
        outputs = modelEnId.generate(input_ids=input_ids, max_length=512)
        translatedSentence = tokenizerEnId.batch_decode(outputs, skip_special_tokens=True)
        return translatedSentence[0]
    elif (originLanguage == 'Sinhala'):
        input_ids = tokenizerEnInc(sentence, return_tensors="pt", padding=True).input_ids
        outputs = modelEnInc.generate(input_ids=input_ids, max_length=512)
        translatedSentence = tokenizerEnInc.batch_decode(outputs, skip_special_tokens=True)
        return translatedSentence[0]


# %%
import speech_recognition as sr
import scipy.io.wavfile

def speech_to_text(audio):
    # audio is a tuple, need to extract the content part
    rate, data = audio

    # Write the audio data to a wav file
    scipy.io.wavfile.write('temp.wav', rate, data)

    r = sr.Recognizer()
    with sr.AudioFile('temp.wav') as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        return text


# %%
def return_answer(user_question, recording, qLanguage, aLanguage):
    # user_question = translateToEng(user_question, qLanguage)
    if user_question:
        print('text is provided')
        user_question = user_question
    else:
        print('voice is provided')
        # If no typed question, then take the voice input, convert it to text
        user_question = speech_to_text(recording)

    q_embedding = model.encode(user_question)

    cos_sim = util.cos_sim(q_embedding, embeddings)

    val = []

    for i in range(len(cos_sim[0])):
        val.append(cos_sim[0][i].item())

    p_answer = answer.copy()
    p_question = question.copy()

    for i in range(len(val)):
        for j in range(i + 1, len(val)):
            if (val[i] < val[j]):
                temp = val[i]
                val[i] = val[j]
                val[j] = temp

                temp = p_answer[i]
                p_answer[i] = p_answer[j]
                p_answer[j] = temp
                temp = p_question[i]
                p_question[i] = p_question[j]
                p_question[j] = temp

    answer1 = "Sorry we don't know the answer"
    answer2 = ""
    answer3 = ""
    if (reranker.compute_score([p_question[0], user_question]) >= -6):
        answer1 = translateFromEng(p_answer[0], aLanguage)
    if (reranker.compute_score([p_question[1], user_question]) >= -6):
        answer2 = translateFromEng(p_answer[1], aLanguage)
    if (reranker.compute_score([p_question[2], user_question]) >= -6):
        answer3 = translateFromEng(p_answer[2], aLanguage)

    return (answer1, answer2, answer3)


translateFromEng("Hello", "Sinhala");

# %%
import gradio as gr  # type: ignore

demo = gr.Interface(
    fn=return_answer,
    inputs=[
        gr.Textbox(label="Typed Question"),
        gr.Audio(label="Voice Question"),
        gr.Dropdown(
            ["English", "Indonesian", "Vietnamese", "Sinhala"], label="Question language", value="English"
        ),
        gr.Dropdown(
            ["English", "Indonesian", "Vietnamese", "Sinhala"], label="Answer language", value="English"
        ),
    ],
    outputs=[gr.Textbox(label="Answer 1"), gr.Textbox(label="Answer 2"), gr.Textbox(label="Answer 3")]
)
demo.launch()