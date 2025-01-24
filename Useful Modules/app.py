import os
os.environ["GOOGLE_API_KEY"] = "<-API_KEY->"

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout, QTextEdit, QLineEdit, QTabWidget, QFrame, QSlider)
from PyQt5.QtGui import QPixmap, QPainter, QFont, QIcon, QTextCursor
from PyQt5.QtCore import Qt, QUrl, QSize
from PyQt5.QtMultimediaWidgets import QVideoWidget
from moviepy import *
import numpy as np
import pyttsx3
import time
import sys

firstTime = True
filePath = ""
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-pro")

class EduDocApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    # UI for application
    def initUI(self):
        self.setWindowTitle('EduDoc')

        # Load background T
        self.background = QPixmap('EduAssets//background.png')

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # EduDoc label
        title = QLabel('EduDoc', self)
        title.setFont(QFont('Arial', 48, QFont.Bold))
        title.setStyleSheet('color: #2476FF; margin-top: 30px; margin-bottom: 30px')
        title.setAlignment(Qt.AlignCenter | Qt.AlignmentFlag.AlignTop)
        main_layout.addWidget(title)

        # File chooser
        file_layout = QHBoxLayout()

        self.doc_label = QLabel('Upload your document', self)
        self.doc_label.setFont(QFont('Arial', 16, QFont.Bold))
        self.doc_label.setStyleSheet('color: black;')
        file_layout.addWidget(self.doc_label)
        
        self.file_button = QPushButton('Choose File', self)
        self.file_button.clicked.connect(self.choose_file)
        self.file_button.setStyleSheet('background-color: #2476FF; border: 2px solid #DEE7FA; border-radius: 30px; padding: 10px')
        file_layout.addWidget(self.file_button)

        self.file_label = QLabel('No file chosen', self)
        self.file_label.setAlignment(Qt.AlignCenter)
        file_layout.addWidget(self.file_label)
        
        # Frame to surround the file_layout
        file_frame = QFrame()
        file_frame.setLayout(file_layout)
        file_frame.setStyleSheet('background-color: #DEE7FA; border: 2px solid #DEE7FA; border-radius: 30px; padding: 10px;')

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
                margin-right: 10px;
                margin-top: 30px;
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
                width: 250px;
                height: 48px;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
        """)
        
        # UI for tabs
        self.first_time = True
        audio_tab = QWidget()
        video_tab = QWidget()
        qdoc_tab = QWidget()
        tabs.addTab(audio_tab, "Audio Generation")
        tabs.addTab(video_tab,  "Video Generation")
        tabs.addTab(qdoc_tab,  "QDocument")
        
        # Audio Generation tab layout
        audio_layout = QVBoxLayout(audio_tab)
        audio_layout.setSpacing(10)
        
        # UI for audio generation button
        self.gen_audio_button = QPushButton('Generate Audio', self)
        self.gen_audio_button.setFont(QFont('Arial', 16, QFont.Bold))
        self.gen_audio_button.setFixedWidth(250)
        self.gen_audio_button.setFixedHeight(80)
        self.gen_audio_button.setStyleSheet('background: #2476FF; color: black; margin-top: 20px')
        self.gen_audio_button.clicked.connect(self.gen_audio)
        audio_layout.addWidget(self.gen_audio_button, alignment=Qt.AlignCenter)

        # Progress Bar
        self.audio_slider = QSlider(Qt.Horizontal, self)
        self.audio_slider.setRange(0, 100)
        self.audio_slider.setFixedWidth(500)
        self.audio_slider.setFixedHeight(40)
        self.audio_slider.sliderMoved.connect(self.set_audio_position)
        audio_layout.addWidget(self.audio_slider, alignment=Qt.AlignCenter)

        self.time_label = QLabel('00:00 / 00:00', self)
        self.time_label.setFont(QFont('Arial', 14, QFont.Bold))
        self.time_label.setFixedWidth(150)
        self.time_label.setAlignment(Qt.AlignCenter)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.play_pause_button = QPushButton()
        self.play_pause_button.setIcon(QIcon('EduAssets//play.png'))  # Play icon
        self.play_pause_button.setStyleSheet('background: white;border: none; margin-left: 250px;')
        self.play_pause_button.setIconSize(QSize(60, 60))
        self.play_pause_button.setGeometry(300, 500, 60, 60)
        self.play_pause_button.clicked.connect(self.play_pause_audio)

        self.restart_button = QPushButton()
        self.restart_button.setIcon(QIcon('EduAssets//restart.png'))  # Restart icon
        self.restart_button.setStyleSheet('background: white;border: none; margin-right: 250px;')
        self.restart_button.setIconSize(QSize(60, 60))
        self.restart_button.setGeometry(380, 500, 60, 60)
        self.restart_button.clicked.connect(self.restart_audio)

        button_layout.addWidget(self.play_pause_button, alignment=Qt.AlignCenter)
        button_layout.addWidget(self.time_label, alignment=Qt.AlignCenter)
        button_layout.addWidget(self.restart_button, alignment=Qt.AlignCenter)
        
        audio_layout.addLayout(button_layout)
        
        self.save_audio_button = QPushButton('Save Audio', self)
        self.save_audio_button.setFont(QFont('Arial', 16, QFont.Bold))
        self.save_audio_button.setStyleSheet('background: #2476FF; color: black; margin-top: 10px')
        self.save_audio_button.setFixedWidth(250)
        self.save_audio_button.setFixedHeight(80)
        self.save_audio_button.clicked.connect(self.save_audio)

        audio_controls_layout = QHBoxLayout()
        audio_controls_layout.addWidget(self.play_pause_button)
        audio_controls_layout.addWidget(self.restart_button)
        audio_layout.addLayout(audio_controls_layout)
        audio_layout.addWidget(self.save_audio_button, alignment=Qt.AlignCenter)

        self.media_player = QMediaPlayer(None, QMediaPlayer.StreamPlayback)
        self.media_player.setNotifyInterval(1000)
        self.media_player.positionChanged.connect(self.update_slider_position)
        self.media_player.durationChanged.connect(self.update_slider_range)

        # Tabs
        main_layout.addWidget(tabs)
        
        # Video Generation tab layout
        video_layout = QVBoxLayout(video_tab)
        video_layout.setSpacing(10)
        
        self.gen_video_button = QPushButton('Generate video', self)
        self.gen_video_button.setFont(QFont('Arial', 16, QFont.Bold))
        self.gen_video_button.setFixedWidth(250)
        self.gen_video_button.setFixedHeight(80)
        self.gen_video_button.setStyleSheet('background: #2476FF; color: black; margin-top: 20px')
        self.gen_video_button.clicked.connect(self.gen_video)
        video_layout.addWidget(self.gen_video_button, alignment=Qt.AlignCenter)
        
        self.video_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video_widget = QVideoWidget()
        video_layout.addWidget(self.video_widget, alignment=Qt.AlignCenter)
        self.video_player.setVideoOutput(self.video_widget)

        self.play_video_button = QPushButton('Play Video', self)
        self.play_video_button.setFont(QFont('Arial', 16, QFont.Bold))
        self.play_video_button.setFixedWidth(250)
        self.play_video_button.setFixedHeight(80)
        self.play_video_button.setStyleSheet('background: #2476FF; color: black; margin-top: 20px')
        self.play_video_button.clicked.connect(self.play_video)
        video_layout.addWidget(self.play_video_button, alignment=Qt.AlignCenter)

        # QDocument tab layout
        qdoc_layout = QVBoxLayout(qdoc_tab)
        qdoc_layout.setContentsMargins(150, 0, 150, 50)
        self.chatbot_output = QTextEdit(self)
        self.chatbot_output.setReadOnly(True)
        
        input_layout = QHBoxLayout()
        self.chatbot_input = QLineEdit(self)
        self.chatbot_input.setPlaceholderText('Ask a question...')
        self.chatbot_input.setFixedHeight(40)
        self.chatbot_input.returnPressed.connect(self.query_document)
        
        self.chatbot_button = QPushButton('Send', self)
        self.chatbot_button.setFixedHeight(40)
        self.chatbot_button.clicked.connect(self.query_document)
        
        input_layout.addWidget(self.chatbot_input)
        input_layout.addWidget(self.chatbot_button)
        
        qdoc_layout.addWidget(self.chatbot_output)
        qdoc_layout.addLayout(input_layout)

        main_layout.addWidget(tabs)
        
        self.showMaximized()

    def choose_file(self):
        global filePath, firstTime
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose Document", "", "All Files (*);;PDF Files (*.pdf)", options=options)
        if file_name:
            self.file_label.setText(file_name)
            filePath = file_name
            firstTime = True
            QApplication.processEvents()
        else:
            self.file_label.setText('No file chosen')
            
    def gen_audio(self):
        global filePath, llm
        loader = PyPDFLoader(filePath)
        pages = loader.load_and_split()
        content = "\n".join([x.page_content for x in pages])
        qa_prompt = "Explain the provided content, mention all the important points, and at last summarize the content. If you don't know the answer, just say that you don't know, don't try to make up an answer."
        input_text = "\nContext:\n" + content + "prompt: "  + qa_prompt
        print("Input Text: ", input_text)
        result = llm.invoke(input_text)
        print("Answer:\n", result.content)
        self.ocr_text = result.content.replace("*", "")
        if self.ocr_text:
            engine = pyttsx3.init()
            audio_path = "generated_audio.wav"
            engine.setProperty('rate', 135)
            engine.save_to_file(self.ocr_text, audio_path)
            engine.runAndWait()
            engine.stop()
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(audio_path)))
            self.play_pause_button.setIcon(QIcon('EduAssets//play.png'))

    def play_pause_audio(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_pause_button.setIcon(QIcon('EduAssets//play.png'))
        else:
            self.media_player.play()
            self.play_pause_button.setIcon(QIcon('EduAssets//pause.png'))

    def restart_audio(self):
        self.media_player.setPosition(0)
        self.media_player.play()
        self.play_pause_button.setIcon(QIcon('EduAssets//pause.png'))

    def set_audio_position(self, position):
        self.media_player.setPosition(position)

    def update_slider_position(self, position):
        self.audio_slider.setValue(position)
        self.update_time()

    def update_slider_range(self, duration):
        self.audio_slider.setRange(0, duration)
        self.update_time()

    def update_time(self):
        current_time = self.media_player.position() // 1000
        total_time = self.media_player.duration() // 1000

        current_minutes = current_time // 60
        current_seconds = current_time % 60
        total_minutes = total_time // 60
        total_seconds = total_time % 60

        self.time_label.setText(f"{current_minutes:02}:{current_seconds:02} / {total_minutes:02}:{total_seconds:02}")

    def save_audio(self):
        options = QFileDialog.Options()
        save_file_name, _ = QFileDialog.getSaveFileName(self, "Save Audio As", "", "WAV Files (*.wav);;All Files (*)", options=options)
        if save_file_name:
            engine = pyttsx3.init()
            engine.setProperty('rate', 135)
            engine.save_to_file(self.ocr_text, save_file_name)
            engine.runAndWait()
            engine.stop()
        
    def gen_video(self):
        self.gen_audio()
        closed_mouth_img_path = 'EduAssets/character_closed.png'  
        open_mouth_img_path = 'EduAssets/character_open.png'     
        audio_path = 'generated_audio.wav'                              
        output_path = 'generated_video.mp4'
        
        audio = AudioFileClip(audio_path)    
        duration = audio.duration
        fps = 8

        img_closed = ImageClip(closed_mouth_img_path).with_duration(0.25)
        img_open = ImageClip(open_mouth_img_path).with_duration(0.125)
        img_closed_op = ImageClip(closed_mouth_img_path).with_duration(0.125)
        
        clips = []
        for t in np.arange(0, duration, 0.25):
            volume = audio.subclipped(t, t + 0.25).to_soundarray().mean()
            if volume > 0:
                clips.append(img_open)
                clips.append(img_closed_op)
            else:
                clips.append(img_closed)
        video = concatenate_videoclips(clips, method='compose')
        video = video.with_audio(audio)
        video.write_videofile(output_path, fps=fps)
        
        print("Video display started")
        self.video_player.setMedia(QMediaContent(QUrl.fromLocalFile(output_path)))
        print("Video display ended")
    
    def play_video(self):
        if self.video_player.state() == QMediaPlayer.PlayingState:
            self.video_player.pause()
        else:
            self.video_player.play()
        
    def add_chat_message(self, text, is_user):
        if is_user:
            self.chatbot_output.append(f"<div style='text-align: left; font: 20px 'Arial'; color: #2476FF; padding: 10px; margin: 5px 20px; border-radius: 10px;'>You:   {text}</div>")
        else:
            self.chatbot_output.append(f"<div style='text-align: left; color: black; font: 20px 'Arial' padding: 10px; margin: 5px 20px; border-radius: 10px;'>Bot:   {text}\n\n</div>")
        self.chatbot_output.moveCursor(QTextCursor.End)

    def query_document(self):
        global firstTime, filePath, embeddings, db, llm
        user_question = self.chatbot_input.text()
        if user_question:
            self.add_chat_message(user_question, True)
            self.chatbot_input.clear()
            QApplication.processEvents()

            if filePath == "":
                self.add_chat_message("Please upload a document first", False)
                return
            
            print("Query Execution Started")
            start = time.time()
            if firstTime:
                loader = PyPDFLoader(filePath)
                pages = loader.load_and_split()
                db = FAISS.from_documents(pages, embeddings)
                firstTime = False

            docs = db.similarity_search(user_question)
            content = "\n".join([x.page_content for x in docs])
            qa_prompt = "Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.----------------"
            input_text = qa_prompt + "\nContext:" + content + "\nUser question:\n" + user_question
            print("context: ", content)
            print("db:\n",db)
            result = llm.invoke(input_text)
            print("Answer: ", result.content)
            end = time.time()
            print("Query Execution took ", end-start, "s")

            self.add_chat_message(result.content, False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.background)
        

def main():
    app = QApplication(sys.argv)
    ex = EduDocApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
