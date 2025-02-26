import os
os.environ["GOOGLE_API_KEY"] = "<-API-KEY->"

import warnings
warnings.filterwarnings("ignore")

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import speech_recognition as sr
import pyttsx3
import uuid
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout,QRadioButton, QListWidgetItem, QListWidget, QPushButton, QLabel, QFileDialog, QHBoxLayout, QTextEdit, QLineEdit, QTabWidget, QFrame, QMainWindow, QSplitter)
from PyQt5.QtGui import QPixmap, QPainter, QFont, QTextCursor, QIcon
from PyQt5.QtCore import Qt, QUrl, QSize
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from moviepy import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips
import numpy as np
import sys
import shutil

online = True
firstTime = True
filePaths = []
agent_names = ["smily", "Omni", "Lucy"]
curr_agent = agent_names[0]
selected_agent = agent_names[0]
voices = ["male", "female"]
selected_voice = voices[0] 
curr_voice = voices[1]
replay_video = QUrl.fromLocalFile(f"assets/characters/{curr_agent}/standing.gif")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-pro")

off_llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q6_K.bin", model_type="llama", 
                        config={'max_new_tokens': 512, 'temperature': 0.01})

recognizer = sr.Recognizer()

def changeDB():
    global filePaths, embeddings, db
    
    pdfs = []
    for filePath in filePaths:
        loader = PyPDFLoader(filePath)
        pages = loader.load_and_split()
        pdfs.extend(pages)
    db = FAISS.from_documents(pdfs, embeddings)
    db.metadata = {"file_paths": filePaths}

class FileItemWidget(QWidget):
    def __init__(self, file_path, remove_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.remove_callback = remove_callback

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.remove_button = QPushButton("X")
        self.remove_button.setStyleSheet("background-color: #ff4d4d; color: white; font-weight: bold; border-radius: 11px;")
        self.remove_button.setFixedSize(22, 22)
        self.remove_button.clicked.connect(self.remove_item)
        layout.addWidget(self.remove_button)
        
        self.label = QLabel(file_path.split('/')[-1])
        layout.addWidget(self.label)
        QApplication.processEvents()
        global filePaths
        filePaths.append(file_path)
        changeDB()

    def remove_item(self):
        self.remove_callback(self.file_path)
        QApplication.processEvents()
        global filePaths
        filePaths.remove(self.file_path)
        changeDB()
        

class DragDropWidget(QListWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setStyleSheet("background-color: #DEE7FA; border: 2px dashed #ccc;")
        self.file_paths = []

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                self.add_file(file_path)
            event.acceptProposedAction()

    def add_file(self, file_path):
        if file_path not in self.file_paths:
            self.file_paths.append(file_path)
            item_widget = FileItemWidget(file_path, self.remove_file)
            item = QListWidgetItem(self)
            item.setSizeHint(item_widget.sizeHint())
            self.addItem(item)
            self.setItemWidget(item, item_widget)

    def remove_file(self, file_path):
        self.file_paths.remove(file_path)
        for index in range(self.count()):
            item = self.item(index)
            item_widget = self.itemWidget(item)
            if item_widget.file_path == file_path:
                self.takeItem(index)
                break
            
class QFlowLayout(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.items = []

    def addWidget(self, widget):
        self.items.append(widget)
        super().addWidget(widget)

    def resizeEvent(self, event):
        max_width = self.parentWidget().width()
        x_offset, y_offset = 0, 0
        row_height = 0
        for widget in self.items:
            widget.adjustSize()
            w, h = widget.sizeHint().width(), widget.sizeHint().height()
            if x_offset + w > max_width:
                x_offset = 0
                y_offset += row_height
                row_height = 0
            widget.move(x_offset, y_offset)
            x_offset += w + 5
            row_height = max(row_height, h)
            
            
class EduDocApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('EduDoc - The Intelligent Document Assistant')

        self.background = QPixmap('assets//background//bg.png')

        self.container = QWidget()
        self.layout = QHBoxLayout(self.container)
        self.layout.setContentsMargins(50, 50, 50, 50)
        
        self.splitter = QSplitter(Qt.Horizontal)
        
        self.left_pane = DragDropWidget()
        left_layout = QVBoxLayout()
        file_label = QLabel("File Manager")
        file_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        left_layout.addWidget(file_label, alignment=Qt.AlignCenter) 
        self.open_file_button = QPushButton("Add File")
        self.open_file_button.setStyleSheet('background-color: #2476FF;border: 2px solid #2476FF; border-radius: 20px; font-weight: semi-bold; padding: 10px; margin-top: 20px; font-size: 14px; color: white;')
        self.open_file_button.clicked.connect(self.openFile)
        left_layout.addWidget(self.open_file_button)
        
        left_layout.addWidget(self.left_pane)

        left_container = QWidget()
        left_container.setMinimumWidth(150)
        left_container.setLayout(left_layout)
        
        self.left_pane.setStyleSheet("background-color: #DEE7FA;  border-radius: 30px; ")
        
        self.folder_button = QPushButton()
        self.folder_button.setIcon(QIcon("assets/Icons/folder.png"))
        self.folder_button.setIconSize(QSize(60, 60))
        self.folder_button.setStyleSheet('border: none; margin-right: 5px')
        self.folder_button.clicked.connect(self.toggle_left_pane)
        
        self.agent_button = QPushButton()
        self.agent_button.setIcon(QIcon("assets/Icons/agent.png"))
        self.agent_button.setIconSize(QSize(90, 90))
        self.agent_button.setStyleSheet('border: none; margin-left: 5px')
        self.agent_button.clicked.connect(self.toggle_right_pane)
        
        self.file_paths = []

        self.splitter.addWidget(left_container)
        
        right_pane_layout = QVBoxLayout()
        
        agent_label = QLabel("Agent Selector")
        agent_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        right_pane_layout.addWidget(agent_label, alignment=Qt.AlignCenter)
        right_pane_layout.addSpacing(30)
        
        right_pane = QWidget()
        right_layout = QVBoxLayout()
        right_pane.setLayout(right_layout)
        right_pane.setStyleSheet("background-color: #DEE7FA; border-radius: 30px;")
            
        right_layout.addSpacing(30)
    
        self.mode_layout = QVBoxLayout()
        self.mode_label = QLabel("Select Mode")
        self.mode_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.online_radio = QRadioButton("Online")
        self.online_radio.setStyleSheet("font-size: 14px;")
        self.offline_radio = QRadioButton("Offline")
        self.offline_radio.setStyleSheet("font-size: 14px;")
        self.online_radio.setChecked(True)
        
        self.mode_layout.addWidget(self.mode_label)
        self.mode_layout.addWidget(self.online_radio)
        self.mode_layout.addWidget(self.offline_radio)
        right_layout.addLayout(self.mode_layout)
        
        selectVoice_label = QLabel("Select Voice")
        selectVoice_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 15px;")
        right_layout.addWidget(selectVoice_label) 
        self.male_radio = QRadioButton("Male")
        self.male_radio.setStyleSheet("font-size: 14px;")
        self.female_radio = QRadioButton("Female")
        self.female_radio.setStyleSheet("font-size: 14px;")
        self.male_radio.setChecked(True)
        
        checkVoice_layout = QHBoxLayout()
        checkVoice_label = QLabel("Check Voice")
        checkVoice_label.setStyleSheet("font-weight: bold;")
        checkVoice_layout.addWidget(checkVoice_label)
        
        self.play_button = QPushButton()
        self.play_button.setIcon(QIcon("assets/Icons/play.png"))
        self.play_button.setIconSize(QSize(24, 24))
        self.play_button.setStyleSheet('border: none; margin-left: 15px;')
        self.play_button.clicked.connect(self.toggle_voice)
        checkVoice_layout.addWidget(self.play_button)
        checkVoice_layout.addStretch()
        
        voice_layout = QVBoxLayout()
        voice_layout.addWidget(self.male_radio)
        voice_layout.addWidget(self.female_radio)
        voice_layout.addSpacing(15)
        voice_layout.addLayout(checkVoice_layout)

        right_layout.addLayout(voice_layout)
        
        right_layout.addSpacing(30)
        self.agent_label = QLabel("Select Agent")
        self.agent_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(self.agent_label)
        
        self.agent_images_layout = QFlowLayout()
        self.load_images()
        right_layout.addLayout(self.agent_images_layout)
        
        right_pane_layout.addWidget(right_pane)
        
        self.save_agent_button = QPushButton("Apply Changes")
        self.save_agent_button.setStyleSheet('background-color: #2476FF;border: 2px solid #2476FF; border-radius: 20px; font-weight: semi-bold; padding: 10px; margin-top: 20px; font-size: 14px; color: white;')
        self.save_agent_button.clicked.connect(self.apply_agents)
        right_pane_layout.addWidget(self.save_agent_button)
        
        right_container = QWidget()
        right_container.setMinimumWidth(150)
        right_container.setMaximumWidth(250)
        right_container.setLayout(right_pane_layout)
        
        right_layout.addStretch()
        
        self.AgentMediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.AgentMediaPlayer.mediaStatusChanged.connect(self.on_agent_media_status_changed)
        
        self.main_pane = QWidget()
        main_layout = QVBoxLayout(self.main_pane)
        self.main_pane.setLayout(main_layout)
        main_layout.setContentsMargins(80, 120, 0, 25)
        main_layout.addStretch()

        # Main Tab
        self.main_tab = QHBoxLayout()

        qdoc_frame_layout = QVBoxLayout()
        qdoc_frame_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        qdoc_frame_layout.setContentsMargins(0, 50, 0, 50)

        qdoc_layout = QVBoxLayout()

        self.chatbot_output = QTextEdit(self)
        self.chatbot_output.setStyleSheet('background: #DEE7FA; color: black; border: none; border-radius: 15px; padding: 10px;')
        self.chatbot_output.setMinimumWidth(600)
        self.chatbot_output.setFixedHeight(620)
        self.chatbot_output.setReadOnly(True)

        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.chatbot_input = QLineEdit(self)
        self.chatbot_input.setPlaceholderText('Ask a question...')
        self.chatbot_input.setFixedHeight(70)
        self.chatbot_input.setMinimumWidth(450)
        self.chatbot_input.setStyleSheet('color: black; padding-left: 15px; border: 2px solid #DEE7FA; border-radius: 20px; background-color: white;  margin-bottom: 10px;')
        self.chatbot_input.returnPressed.connect(self.Reply)

        self.chatbot_button = QPushButton('Send', self)
        self.chatbot_button.setFixedHeight(70)
        self.chatbot_button.setFixedWidth(100)
        self.chatbot_button.setStyleSheet('background: #2476FF; color: white; border: none; border-radius: 20px; font-weight: bold; margin-bottom: 10px;')
        self.chatbot_button.clicked.connect(self.Reply)

        input_layout.addWidget(self.chatbot_input)
        input_layout.addWidget(self.chatbot_button)

        qdoc_layout.addWidget(self.chatbot_output)
        qdoc_layout.addSpacing(20)
        qdoc_layout.addLayout(input_layout)

        qdoc_frame = QFrame()
        qdoc_frame.setLayout(qdoc_layout)
        qdoc_frame.setStyleSheet('background-color: #DEE7FA; border: 2px solid #DEE7FA; border-radius: 15px; padding: 15px; ')

        qdoc_frame_layout.addWidget(qdoc_frame)
        
        face_layout = QVBoxLayout()
        face_layout.setContentsMargins(0, 250, 0, 250)
        
        self.face_image = QLabel(self)
        self.face_image.setPixmap(QPixmap(f'assets/characters/{curr_agent}/char.png').scaled(QSize(250, 250), Qt.KeepAspectRatio))
        self.face_image.setFixedWidth(250)
        self.face_image.setFixedHeight(250)
        self.face_image.setAlignment(Qt.AlignCenter)
        
        face_layout.addWidget(self.face_image,alignment=Qt.AlignCenter)
        self.face_image.show()
        
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()
        face_layout.addWidget(self.videoWidget, alignment=Qt.AlignCenter)
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.videoWidget.setFixedSize(250, 250)

        self.videoWidget.hide()

        self.mediaPlayer.mediaStatusChanged.connect(self.on_media_status_changed)
    
        self.main_tab.addLayout(qdoc_frame_layout)
        self.main_tab.addLayout(face_layout)
        main_layout.addLayout(self.main_tab)
        self.splitter.addWidget(self.main_pane)
        self.splitter.addWidget(right_container)
        self.layout.addWidget(self.folder_button)
        self.layout.addWidget(self.splitter)
        self.layout.addWidget(self.agent_button)
        
        self.setCentralWidget(self.container)
        self.splitter.setSizes([0, 1000, 0])
        self.splitter.setHandleWidth(5)
        self.splitter.setStyleSheet("QSplitter::handle {background-color: #DEE7FA; height: 100px; width: 5px;}")
        
        self.showMaximized()
        
    def paintEvent(self, _):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.background)
    
    def toggle_left_pane(self):
        size = self.splitter.sizes()
        if size[0] == 0:
            self.splitter.setSizes([400, size[1], size[2]])
        else:
            self.splitter.setSizes([0, size[1], size[2]])

    def toggle_right_pane(self):
        size = self.splitter.sizes()
        if size[2] == 0:
            self.splitter.setSizes([size[0], size[1], 250])
        else:
            self.splitter.setSizes([size[0], size[1], 0])

    def openFile(self):
        options = QFileDialog.Options()
        file_names, _ = QFileDialog.getOpenFileNames(self, "Open File", "", "PDF Files (*.pdf)", options=options)
        for file_path in file_names:
            self.left_pane.add_file(file_path)


    def addFileToList(self, file_path):
        if file_path not in self.file_paths:
            item = QListWidgetItem("- " + file_path.split("/")[-1])
            self.file_paths.append(file_path)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.file_list.itemClicked.connect(self.removeFileFromList)

    def removeFileFromList(self, item):
        row = self.file_list.row(item)
        self.file_list.takeItem(row)
        file_name = item.text()[2:]
        self.file_paths = [path for path in self.file_paths if not path.endswith(file_name)]
        
    def toggle_voice(self):
            global selected_voice
            if self.AgentMediaPlayer.state() == QMediaPlayer.PlayingState:
                self.AgentMediaPlayer.stop()
                self.play_button.setIcon(QIcon("assets/Icons/play.png"))
            else:
                if self.male_radio.isChecked(): 
                    voice_file = "assets/voices/male.wav"
                    selected_voice = "male" 
                elif self.female_radio.isChecked():
                    voice_file = "assets/voices/female.wav"
                    selected_voice = "female"
                else:
                    return
                
                self.AgentMediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(voice_file)))
                self.AgentMediaPlayer.play()
                self.play_button.setIcon(QIcon("assets/Icons/stop.png"))

    def on_agent_media_status_changed(self, status):
        if status == QMediaPlayer.EndOfMedia or status == QMediaPlayer.StoppedState:
            self.play_button.setIcon(QIcon("assets/Icons/play.png"))
            
    def load_images(self):
        global agent_names, curr_agent
        for name in agent_names:
            lbl = QLabel(self)
            pixmap = QPixmap(f"assets/characters/{name}/char.png").scaled(QSize(100, 100), Qt.KeepAspectRatio)
            lbl.setPixmap(pixmap)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(100, 100)
            lbl.setStyleSheet("border: 2px solid; background-color: white; border-radius: 0px;")  
            lbl.mousePressEvent = lambda event, lbl=lbl, name=name: self.select_character(lbl, name)
            if(name == curr_agent):
                self.select_character(lbl, name)
            self.agent_images_layout.addWidget(lbl)

    def select_character(self, lbl, name):
        global selected_agent
        for widget in self.agent_images_layout.items:
            widget.setStyleSheet("border: 2px solid; background-color: white; border-radius: 0px;")
        lbl.setStyleSheet("border: 2px solid blue; background-color: white; border-radius: 5px;")
        selected_agent = name
    
    def apply_agents(self):
        global curr_agent, selected_agent, selected_voice, curr_voice, replay_video, online
        curr_agent = selected_agent
        curr_voice = selected_voice
        
        if self.online_radio.isChecked():
            online = True
        elif self.offline_radio.isChecked():
            online = False
        
        print(online)
        
        if(curr_agent == "smily"):
            self.face_image.setPixmap(QPixmap(f"assets/characters/{curr_agent}/char.png").scaled(QSize(250, 250), Qt.KeepAspectRatio))
            self.face_image.show()
            self.videoWidget.setFixedSize(250, 250)
            self.videoWidget.hide()
            replay_video = None
        else:
            self.face_image.hide()
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(f"assets/characters/{curr_agent}/standing.gif")))
            replay_video = QUrl.fromLocalFile(f"assets/characters/{curr_agent}/standing.gif")
            self.videoWidget.setFixedSize(250, 400)
            self.videoWidget.show()
            self.mediaPlayer.play()
            
        self.AgentMediaPlayer.stop()
        self.play_button.setIcon(QIcon("assets/Icons/play.png"))
        
        size = self.splitter.sizes()
        self.splitter.setSizes([size[0], size[1], 0])        
            
    def offAI(self, question):
        global off_llm, db, filePaths
        
        if filePaths == []:
            return "0", "Hey! You did not upload your documents. Could you please upload documents?"
        
        print("Query Execution Started in offline mode")
        start = time.time()
        
        qa_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer clear, concise, and informative and neither too short nor too long.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
        """
        
        prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
        
        dbqa = RetrievalQA.from_chain_type(llm=off_llm,
                                    chain_type='stuff',
                                    retriever=db.as_retriever(search_kwargs={'k': 2}),
                                    return_source_documents=True,
                                    chain_type_kwargs={'prompt': prompt}
                                    )
        
        response = dbqa.invoke({'query': question}) 
        end = time.time()
        print("Time taken: ", end-start)
        print(response)
        return "question", response["result"]
    
    def askAI(self, question):
        global db, llm, filePaths
        
        print("Query Execution Started")
        start = time.time()
        
        qa_prompt = '''Your name is Doc AI, and you are an AI assistant that helps users find information based on a given context.  
            You are provided with context or information, and users will ask questions based on that information.  

            You have three tasks:  
            1. Identify if the prompt is a question or a greeting. Return "question" or "greeting" or "garbage" on the first line.  
            
            2. Provide an answer:  
            - If the prompt is a greeting, return an appropriate greeting.  
            - If the prompt is a question, find the answer from the provided context and return:  
                **"According to information provided, (answer)"**  
            - If the answer is not found in the context, return:  
                **"Answer not found in the information provided."**  
            
            3. In next line, 
            - If the answer not found in the context, return the answer from the real-world facts and return:
                **"According to real-world facts, (answer)"**
            - If the answer cannot be found then return as it is.
            - If the answer is found in the context and can be answered from real-world facts,
                and both answers are same, just return the answer from the context.
            - If the answer contradicts, return the answer from the real-world facts in addition to the answer from the context.
            
            Ensure responses are clear, concise, and informative. neither too short nor too long.
        '''
        
        qa_prompt2 = '''Your name is Doc AI, and you are an AI assistant that helps users find information.    

            You have two tasks:  
            1. Identify if the prompt is a question or a greeting. Return "question" or "greeting" or "garbage" on the first line.  
            
            2. Provide an answer:  
            - If the prompt is a greeting, return an appropriate greeting.  
            - If the prompt is a question, find the answer from real-world facts and return:
                **"According to real-world facts, (answer)"**
            - If the answer cannot be found then return 
                **"Answer not found in real-world facts. Did you forgot to upload your documents?"**
            
            Ensure responses are clear, concise, and informative. neither too short nor too long.
        '''
        
        if filePaths == []:
            input_text = qa_prompt2 + "\nPrompt:\n" + question
            result = llm.invoke(input_text)
        else:
            docs = db.similarity_search(question)
            content = "\n".join([x.page_content for x in docs])
            input_text = qa_prompt + "\nInformation: \n" + content + "\nPrompt:\n" + question
            result = llm.invoke(input_text)
        
        answer = result.content
        print("Answer: ", answer)
        
        first_line, *remaining_lines = answer.split('\n', 1)
        print("First Line: ", first_line)
        if remaining_lines:
            print("Remaining: ", remaining_lines[0])
        end = time.time()
        print("Time taken: ", end-start)
        if remaining_lines:
            return first_line, remaining_lines[0]
        else:
            return first_line, " "

    def add_chat_message(self, text, is_user):
        if is_user:
            self.chatbot_output.append(f"<div style='text-align: left; font-family: Arial; font-size: 20px; color: #2476FF; padding: 10px; margin-left: 100px; margin-right: 100px; margin-bottom: 20px; border-radius: 10px;'>You: {text}</div>")
        else:
            self.chatbot_output.append(f"<div style='text-align: left; font-family: Arial; font-size: 20px; color: black; padding: 10px; margin-left: 100px; margin-right: 100px; margin-bottom: 30px; border-radius: 10px;'>Bot: {text}</div>")
        self.chatbot_output.moveCursor(QTextCursor.End)
    
    def on_media_status_changed(self, status):
            if status == QMediaPlayer.EndOfMedia and curr_agent == "smily":
                self.videoWidget.hide()
                self.face_image.show()
            elif status == QMediaPlayer.EndOfMedia:
                self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(f"assets/characters/{curr_agent}/standing.gif")))
                self.mediaPlayer.play()
                
    def generate_speaking_video(self, typ, audio_path, output_path):
        global curr_agent
        audio = AudioFileClip(audio_path)
        audio_duration = audio.duration
        fps = 24
        
        if curr_agent == "smily":
            img_closed = ImageClip("assets/characters/smily/character_closed.png").with_duration(0.30)
            img_open = ImageClip("assets/characters/smily/character_open.png").with_duration(0.15)
            img_closed_op = ImageClip("assets/characters/smily/character_closed.png").with_duration(0.15)
            clips = []
            for t in np.arange(0, audio_duration, 0.30):
                volume = audio.subclipped(t, t + 0.30).to_soundarray().mean()
                if volume > 0:
                    clips.append(img_open)
                    clips.append(img_closed_op)
                else:
                    clips.append(img_closed)
            video = concatenate_videoclips(clips, method='compose')
        else:
            if typ == "question" or typ == "0":
                gif_clip = VideoFileClip(f"assets/characters/{curr_agent}/explaining.gif").with_duration(audio_duration)
                video = gif_clip
                gif_duration = gif_clip.duration
            else:
                gif_clip = VideoFileClip(f"assets/characters/{curr_agent}/hello.gif").with_duration(audio_duration)
                video = gif_clip
                
            gif_duration = gif_clip.duration
        
            # Calculate the number of times the gif needs to be repeated
            repeat_count = int(audio_duration // gif_duration) + 1
            repeated_gif_clip = concatenate_videoclips([gif_clip] * repeat_count)
            repeated_gif_clip = repeated_gif_clip.with_duration(audio_duration)

            video_clip = repeated_gif_clip.with_audio(audio)
            video_clip.write_videofile(output_path, codec='libx264')
            
        video = video.with_audio(audio)
        video.write_videofile(output_path, fps=fps)

    def Reply(self):
        global curr_agent, curr_voice, replay_video, online
        
        user_question = self.chatbot_input.text()
        if user_question:
            self.add_chat_message(user_question, True)
            self.chatbot_input.clear()
            QApplication.processEvents()
            if online:
                typ, answer = self.askAI(user_question)
            else:
                typ, answer = self.offAI(user_question)
            
            if curr_agent!= "smily":
                self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(f"assets/characters/{curr_agent}/thinking.gif")))
                self.mediaPlayer.play()
                
            if(answer == ' ' or typ=="garbage"):
                answer = "Sorry! I can't understand this question"
            
            engine = pyttsx3.init()
            engine.setProperty('rate', 140)
            voices = engine.getProperty('voices')
            if curr_voice == "male":
                engine.setProperty('voice', voices[0].id)
            else:
                engine.setProperty('voice', voices[1].id)

            audio_filename = f"tmp/audio_{uuid.uuid4().hex}.wav"
            video_filename = f"tmp/video_{uuid.uuid4().hex}.mp4"

            engine.save_to_file(answer, audio_filename)
            engine.runAndWait()

            self.generate_speaking_video(typ, audio_filename, video_filename)
            
            QApplication.processEvents()
            video_url = QUrl.fromLocalFile(video_filename)
            
            QApplication.processEvents()
            if curr_agent == "smily":
                self.face_image.hide()
                self.videoWidget.show()
                
            replay_video = video_url
            
            self.mediaPlayer.setMedia(QMediaContent(video_url))
            self.add_chat_message(answer, False)
            self.mediaPlayer.play()

def main():
    tmp_folder = 'tmp'
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)
    app = QApplication(sys.argv)
    ex = EduDocApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
