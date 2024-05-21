from sentence_transformers import SentenceTransformer, util
#%%
import pandas as pd

df = pd.read_csv('data.csv')

question = df['Question']
answer = df['Answer']

#%%
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#%%
embeddings = model.encode(question)
#%%
# Compute cosine similarity between all pairs
cos_sim = util.cos_sim(embeddings, embeddings)

# Add all pairs to a list with their cosine similarity score
all_sentence_combinations = []
for i in range(len(cos_sim) - 1):
    for j in range(i + 1, len(cos_sim)):
        all_sentence_combinations.append([cos_sim[i][j], i, j])
#%%
# Sort list by the highest cosine similarity score
all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)
#%%
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

#%%
import speech_recognition as sr

def voice_to_text():
    # Initialize recognizer class (for recognizing the speech)
    r = sr.Recognizer()

    # Use Microphone to input speech
    with sr.Microphone() as source:
        print("Listening for question...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except:
            print("Could not recognize your voice. Please try again.")
            return None
#%%
def get_user_question(clicked_button):
    # global clicked_button
    if clicked_button == 'button1':
        return entry.get()
    elif clicked_button == 'button2':
        return voice_to_text()
    else:
        return
#%%
def change_text_helper(clicked_button):
    user_question = get_user_question(clicked_button)  # get user's question by voice
    if user_question is None:
        return None

    q_embedding = model.encode(user_question)
    cos_sim = util.cos_sim(q_embedding, embeddings)

    val = [cos_sim[0][i].item() for i in range(len(cos_sim[0]))]
    p_answer = answer.copy()
    p_question = question.copy()

    val, p_answer, p_question = zip(*sorted(zip(val, p_answer, p_question), reverse=True))

    if reranker.compute_score([p_question[0], user_question]) > 0:
        return p_answer[0]
    else:
        return "Sorry, we don't know the answer for this question"
#%%
def change_text2():
    result = change_text_helper('button2')

    if result is None:
        answer_label.config(text="Please ask your question again.")
    else:
        answer_label.config(text=result)
#%%

def change_text1():
    result = change_text_helper('button1')
    answer_label.config(text=result if result is not None else "Sorry, we don't know the answer for this question")
#%%
import tkinter as tk

window = tk.Tk()
window.geometry('520x300')
label = tk.Label(text="FAQ")
label.pack()

entry = tk.Entry(fg="black", bg="white", width=50)
entry.pack()

# Create a button
button1 = tk.Button(window, text="Find", command=change_text1)
button1.pack()

button2 = tk.Button(window, text="Speak Question", command=change_text2)
button2.pack()

noti_label = tk.Label(text="Answer")
noti_label.pack()

answer_label = tk.Label(text="", wraplength=300, justify='center')
answer_label.pack()

# Run the main event loop
window.mainloop()