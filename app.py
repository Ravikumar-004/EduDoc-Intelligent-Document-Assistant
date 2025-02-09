import os
os.environ["GOOGLE_API_KEY"] = "API_KEY"

import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout, QTextEdit, QLineEdit, QTabWidget, QFrame, QMainWindow)
from PyQt5.QtGui import QPixmap, QPainter, QFont, QTextCursor
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import speech_recognition as sr
from moviepy import AudioFileClip, ImageClip, concatenate_videoclips
import numpy as np
import pyttsx3
import time
import sys
import uuid

firstTime = True
filePath = ""
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-pro")
recognizer = sr.Recognizer()

class EduDocApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    # UI for application
    def initUI(self):
        self.setWindowTitle('EduDoc - The Intelligent Document Assistant')

        # Load background
        self.background = QPixmap('EduAssets//background.png')

        # Main layout
        main_layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        main_layout.setContentsMargins(0,80,0,0)

        # File chooser
        file_layout = QHBoxLayout()

        self.doc_label = QLabel('Upload your document', self)
        self.doc_label.setFont(QFont('Arial', 16, QFont.Bold))
        self.doc_label.setStyleSheet('color: black;')
        file_layout.addWidget(self.doc_label)
        
        self.file_button = QPushButton('Choose File', self)
        self.file_button.clicked.connect(self.choose_file)
        self.file_button.setStyleSheet('background: #1cac78; color: black; border: none; border-radius: 30px; padding: 15px; font-size: 18px;')
        file_layout.addWidget(self.file_button)

        self.file_label = QLabel('No file chosen', self)
        self.file_label.setAlignment(Qt.AlignCenter)
        file_layout.addWidget(self.file_label)
        
        # Frame to surround the file_layout
        file_frame = QFrame()
        file_frame.setLayout(file_layout)
        file_frame.setStyleSheet('background-color: #00C375; border: 2px solid #00C375; border-radius: 30px; padding: 10px; padding-left: 25px; padding-right: 25px; ')

        main_layout.addWidget(file_frame, alignment=Qt.AlignHCenter)

        # Tabs
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
            }
            QTabBar::tab {
                background: white;
                color: #000000;
                font: bold 28px 'Arial';
                margin-right: 20px;
                margin-top: 50px;
                margin-bottom: 30px;
            }
            QTabBar::tab:selected {
                color: #2476FF;
                border-bottom: 2px solid #2476FF;
            }
            QTabBar::tab:hover {
                color: #195BB5;
            }
            QTabBar::tab:selected:hover {
                color: #195BB5;
                border-bottom: 2px solid #195BB5;
            }
            QTabBar::tab {
                width: 300px;
                height: 48px;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
        """)
        
        # UI for tabs
        self.first_time = True
        qdoc_tab = QWidget()
        tabs.addTab(qdoc_tab,  "QDocument")
        
        qdoc_frame_layout = QVBoxLayout(qdoc_tab)
        qdoc_frame_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        qdoc_frame_layout.setContentsMargins(150, 50, 150, 50)

        qdoc_layout = QVBoxLayout()

        self.chatbot_output = QTextEdit(self)
        self.chatbot_output.setStyleSheet('background: #DEE7FA; color: black; border: none; border-radius: 15px; padding: 10px;')
        self.chatbot_output.setFixedWidth(1000)
        self.chatbot_output.setFixedHeight(600)
        self.chatbot_output.setReadOnly(True)

        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.chatbot_input = QLineEdit(self)
        self.chatbot_input.setPlaceholderText('Ask a question...')
        self.chatbot_input.setFixedHeight(80)
        self.chatbot_input.setFixedWidth(1050)
        self.chatbot_input.setStyleSheet('color: black; padding-left: 15px; border: 2px solid #DEE7FA; border-radius: 20px; background-color: white;  margin-bottom: 10px;')
        self.chatbot_input.returnPressed.connect(self.query_document)

        self.chatbot_button = QPushButton('Send', self)
        self.chatbot_button.setFixedHeight(70)
        self.chatbot_button.setFixedWidth(120)
        self.chatbot_button.setStyleSheet('background: #2476FF; color: white; border: none; border-radius: 20px; font-weight: bold; margin-bottom: 10px;')
        self.chatbot_button.clicked.connect(self.query_document)

        input_layout.addWidget(self.chatbot_input)
        input_layout.addWidget(self.chatbot_button)

        qdoc_layout.addWidget(self.chatbot_output)
        qdoc_layout.addSpacing(20)
        qdoc_layout.addLayout(input_layout)

        qdoc_frame = QFrame()
        qdoc_frame.setLayout(qdoc_layout)
        qdoc_frame.setStyleSheet('background-color: #DEE7FA; border: 2px solid #DEE7FA; border-radius: 15px; padding: 15px; margin-left: 150px; margin-right: 150px; margin-bottom: 80px;')

        qdoc_frame_layout.addWidget(qdoc_frame)
        
        # Face to Face tab
        face_tab = QWidget()
        tabs.addTab(face_tab, "Face to Face")
        
        face_layout = QVBoxLayout(face_tab)
        face_layout.setContentsMargins(0, 0, 0, 200)
        face_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        
        self.listening_label = QLabel('', self)
        self.listening_label.setFont(QFont('Arial', 16, QFont.Bold))
        self.listening_label.setStyleSheet('color: red;margin-top: 30px;')
        self.listening_label.setAlignment(Qt.AlignCenter)
        self.listening_label.setText('Press K and speak')

        face_layout.addWidget(self.listening_label)
        
        self.face_image = QLabel(self)
        self.face_image.setPixmap(QPixmap('EduAssets/character_closed.png'))
        self.face_image.setFixedWidth(600)
        self.face_image.setFixedHeight(600)
        self.face_image.setAlignment(Qt.AlignCenter)
        
        face_layout.addWidget(self.face_image)
        
        
        # Video player setup
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()
        face_layout.addWidget(self.videoWidget)
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.videoWidget.setFixedSize(600, 600)  # Set the size of the video widget
        self.videoWidget.hide()  # Initially hide the video widget

        main_layout.addWidget(tabs)
        
        self.showMaximized()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.background)
            
    def choose_file(self):
        global filePath
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose Document", "", "PDF Files (*.pdf)", options=options)
        if file_name:
            self.file_label.setText(file_name)
            filePath = file_name
            QApplication.processEvents()
            loader = PyPDFLoader(filePath)
            pages = loader.load_and_split()
            global embeddings, db
            db = FAISS.from_documents(pages, embeddings)
        else:
            self.file_label.setText('No file chosen')
            
    def query_document(self):
        user_question = self.chatbot_input.text()
        if user_question:
            self.add_chat_message(user_question, True)
            self.chatbot_input.clear()
            QApplication.processEvents()
            answer = self.askAI(user_question)
            self.add_chat_message(answer, False)
            
    def askAI(self, question):
        global db, llm, filePath
        if filePath == "":
            return "Please upload a document first"
        print("Query Execution Started")
        start = time.time()
        docs = db.similarity_search(question)
        content = "\n".join([x.page_content for x in docs])
        qa_prompt = "Use the following pieces of context to answer the user's question. Eloborate the answer, do not give 1 word answers. If you don't know the answer, just say that you don't know, don't try to make up an answer.----------------"
        input_text = qa_prompt + "\nContext:" + content + "\nUser question:\n" + question
        result = llm.invoke(input_text)
        print("Answer: ", result.content)
        end = time.time()
        print("Time taken: ", end-start)
        return result.content

    def add_chat_message(self, text, is_user):
        if is_user:
            self.chatbot_output.append(f"<div style='text-align: left; font: 20px 'Arial'; color: #2476FF; padding: 10px; margin-left: 100px; margin-bottom: 20px;border-radius: 10px;'>You:   {text}</div>")
        else:
            self.chatbot_output.append(f"<div style='text-align: left; color: black; font: 20px 'Arial' padding: 10px; margin-left: 100px 20px; margin-bottom: 30px; border-radius: 10px;'>Bot:   {text}</div>")
        self.chatbot_output.moveCursor(QTextCursor.End)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_K:
            self.listening_label.setText("Listening...")
            QApplication.processEvents()
            result,c = self.listen_to_speech()
            QApplication.processEvents()
            self.listening_label.setText(result)
            self.Reply(result, c)
        
        if event.key() == Qt.Key_R:
            self.face_image.hide()
            self.videoWidget.show()
            self.play()
            
    def listen_to_speech(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                print("You said: " + text)
                return text,1
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                return "Can't understand audio.",0
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
                return "Error! Error!! Error in Google Speech Recognition service",0

    def generate_speaking_video(self, closed_mouth_img_path, open_mouth_img_path, audio_path, output_path):
        audio = AudioFileClip(audio_path)
        duration = audio.duration
        fps = 24
        img_closed = ImageClip(closed_mouth_img_path).with_duration(0.30)
        img_open = ImageClip(open_mouth_img_path).with_duration(0.15)
        img_closed_op = ImageClip(closed_mouth_img_path).with_duration(0.15)
        clips = []
        for t in np.arange(0, duration, 0.30):
            volume = audio.subclipped(t, t + 0.30).to_soundarray().mean()
            if volume > 0:
                clips.append(img_open)
                clips.append(img_closed_op)
            else:
                clips.append(img_closed)
        video = concatenate_videoclips(clips, method='compose')
        video = video.with_audio(audio)
        video.write_videofile(output_path, fps=fps)

      # Add this import at the beginning

    def Reply(self, result, c):
        self.listening_label.setText("Wait a moment...") 
        if c:
            answer = self.askAI(result)
        else:
            answer = result

        engine = pyttsx3.init()
        engine.setProperty('rate', 140)

        # Generate a unique filename for each query
        audio_filename = f"tmp/audio_{uuid.uuid4().hex}.wav"
        video_filename = f"tmp/video_{uuid.uuid4().hex}.mp4"

        engine.save_to_file(answer, audio_filename)
        engine.runAndWait()

        self.generate_speaking_video('EduAssets/character_closed.png', 
                                    'EduAssets/character_open.png', 
                                    audio_filename, 
                                    video_filename)

        QApplication.processEvents()
        self.listening_label.setText("...") 

        video_url = QUrl.fromLocalFile(video_filename)
        self.mediaPlayer.setMedia(QMediaContent(video_url))

        self.face_image.hide()
        self.videoWidget.show()
        self.mediaPlayer.play()

        if self.mediaPlayer.state() == QMediaPlayer.StoppedState:
            self.listening_label.setText("Press K to Speak")
            self.videoWidget.hide()
            self.face_image.show()

        
    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

def main():
    app = QApplication(sys.argv)
    ex = EduDocApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()