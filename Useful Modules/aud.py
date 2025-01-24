import os
os.environ["GOOGLE_API_KEY"] = "<-API_KEY->"

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import pyttsx3

def text_to_speech(text, audio_file):
    engine = pyttsx3.init()
    engine.setProperty('rate', 140)
    engine.save_to_file(text, audio_file)
    engine.runAndWait()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-pro")

loader = PyPDFLoader("ProjectStatusLetter1-EduWave.pdf")
pages = loader.load_and_split()

# db = FAISS.from_documents(pages, embeddings)
# docs = db.similarity_search(user_question)

content = "\n".join([x.page_content for x in pages])
qa_prompt = "Explain the provided content, mention all the important points, and at last summarize the content. If you don't know the answer, just say that you don't know, don't try to make up an answer."
input_text = "\nContext:\n" + content + "prompt: "  + qa_prompt
print("Input Text: ", input_text)
result = llm.invoke(input_text)
print("Answer:\n", result.content)

text_to_speech(result.content, "test_aud.wav")

# test_aud = "Hello, This is Edu Duc Assistant. I am here to help you with your queries. Please feel free to ask me anything. If I don't know the answer, I will let you know. Let's get started."
# text_to_speech(test_aud, "test_aud.wav")
