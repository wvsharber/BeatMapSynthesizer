# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BMS_gui.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import beatmapsynth
import io
from contextlib import redirect_stdout

class Ui_BeatMapSynth_GUI(object):
    
    def __init__(self):
        self.difficulty = "easy"
        self.model = "rate_modulated_segmented_HMM"
        self.input_filepath = ""
        self.output_name = ""
        self.k_value = 5
        self.version = 2
    
    def setupUi(self, BeatMapSynth_GUI):
        #Window Setup
        BeatMapSynth_GUI.setObjectName("BeatMapSynth_GUI")
        BeatMapSynth_GUI.resize(459, 475)
        self.centralwidget = QtWidgets.QWidget(BeatMapSynth_GUI)
        self.centralwidget.setObjectName("centralwidget")
        #Input Header
        self.input_header = QtWidgets.QLabel(self.centralwidget)
        self.input_header.setGeometry(QtCore.QRect(10, 10, 71, 16))
        header_font = QtGui.QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header_font.setWeight(75)
        self.input_header.setFont(header_font)
        self.input_header.setObjectName("input_header")
        #Output Header
        self.output_header = QtWidgets.QLabel(self.centralwidget)
        self.output_header.setGeometry(QtCore.QRect(10, 80, 101, 16))
        self.output_header.setFont(header_font)
        self.output_header.setObjectName("output_header")
        #Required parameters header
        self.req_params_label = QtWidgets.QLabel(self.centralwidget)
        self.req_params_label.setGeometry(QtCore.QRect(10, 160, 151, 16))
        self.req_params_label.setFont(header_font)
        self.req_params_label.setObjectName("req_params_label")
        #Optional parameters header
        self.opt_params_label = QtWidgets.QLabel(self.centralwidget)
        self.opt_params_label.setGeometry(QtCore.QRect(10, 310, 151, 16))
        self.opt_params_label.setFont(header_font)
        self.opt_params_label.setObjectName("opt_params_label")
        #Accepted formats label
        self.formats_label = QtWidgets.QLabel(self.centralwidget)
        self.formats_label.setGeometry(QtCore.QRect(10, 30, 301, 16))
        small_font = QtGui.QFont()
        small_font.setPointSize(10)
        self.formats_label.setFont(small_font)
        self.formats_label.setObjectName("formats_label")
        #Output description label
        self.output_label = QtWidgets.QLabel(self.centralwidget)
        self.output_label.setGeometry(QtCore.QRect(10, 100, 391, 21))
        self.output_label.setFont(small_font)
        self.output_label.setWordWrap(True)
        self.output_label.setObjectName("output_label")
        #Output textbox for entry
        self.output_textbox = QtWidgets.QLineEdit(self.centralwidget)
        self.output_textbox.setGeometry(QtCore.QRect(10, 130, 231, 21))
        self.output_textbox.setFont(small_font)
        self.output_textbox.setObjectName("output_textbox")
        #Execute button
        self.execute_button = QtWidgets.QPushButton(self.centralwidget)
        self.execute_button.setGeometry(QtCore.QRect(10, 420, 161, 32))
        self.execute_button.setObjectName("execute_button")
        self.execute_button.clicked.connect(self.execute)
        ##Difficulty group box
        self.difficulty_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.difficulty_groupBox.setGeometry(QtCore.QRect(10, 180, 111, 121))
        self.difficulty_groupBox.setObjectName("difficulty_groupBox")
        #Easy button
        self.easy_radioButton = QtWidgets.QRadioButton(self.difficulty_groupBox)
        self.easy_radioButton.setGeometry(QtCore.QRect(10, 20, 100, 20))
        self.easy_radioButton.setChecked(True)
        self.easy_radioButton.setAutoExclusive(True)
        self.easy_radioButton.setObjectName("easy_radioButton")
        self.easy_radioButton.toggled.connect(self.difficulty_radioButton_selected)
        #Normal Button
        self.normal_radioButton = QtWidgets.QRadioButton(self.difficulty_groupBox)
        self.normal_radioButton.setGeometry(QtCore.QRect(10, 40, 100, 20))
        self.normal_radioButton.setObjectName("normal_radioButton")
        self.normal_radioButton.toggled.connect(self.difficulty_radioButton_selected)
        #Hard Button
        self.hard_radioButton = QtWidgets.QRadioButton(self.difficulty_groupBox)
        self.hard_radioButton.setGeometry(QtCore.QRect(10, 60, 100, 20))
        self.hard_radioButton.setObjectName("hard_radioButton")
        self.hard_radioButton.toggled.connect(self.difficulty_radioButton_selected)
        #Expert Button
        self.expert_radioButton = QtWidgets.QRadioButton(self.difficulty_groupBox)
        self.expert_radioButton.setGeometry(QtCore.QRect(10, 80, 100, 20))
        self.expert_radioButton.setObjectName("expert_radioButton")
        self.expert_radioButton.toggled.connect(self.difficulty_radioButton_selected)
        #ExpertPlus Button
        self.expertplus_radioButton = QtWidgets.QRadioButton(self.difficulty_groupBox)
        self.expertplus_radioButton.setGeometry(QtCore.QRect(10, 100, 100, 20))
        self.expertplus_radioButton.setObjectName("expertplus_radioButton")
        self.expertplus_radioButton.toggled.connect(self.difficulty_radioButton_selected)
        ##Model choice group box
        self.model_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.model_groupBox.setGeometry(QtCore.QRect(150, 180, 241, 101))
        self.model_groupBox.setObjectName("model_groupBox")
        #Random button
        self.random_radioButton = QtWidgets.QRadioButton(self.model_groupBox)
        self.random_radioButton.setGeometry(QtCore.QRect(10, 20, 100, 20))
        self.random_radioButton.setObjectName("random_radioButton")
        self.random_radioButton.toggled.connect(self.model_radioButton_selected)
        #HMM button
        self.HMM_radioButton = QtWidgets.QRadioButton(self.model_groupBox)
        self.HMM_radioButton.setGeometry(QtCore.QRect(10, 40, 100, 20))
        self.HMM_radioButton.setObjectName("HMM_radioButton")
        self.HMM_radioButton.toggled.connect(self.model_radioButton_selected)
        #Segmented HMM button
        self.segHMM_radioButton = QtWidgets.QRadioButton(self.model_groupBox)
        self.segHMM_radioButton.setGeometry(QtCore.QRect(10, 60, 131, 21))
        self.segHMM_radioButton.setObjectName("segHMM_radioButton")
        self.segHMM_radioButton.toggled.connect(self.model_radioButton_selected)
        #Rate modulated segmented HMM button
        self.ratesegHMM_radioButton = QtWidgets.QRadioButton(self.model_groupBox)
        self.ratesegHMM_radioButton.setGeometry(QtCore.QRect(10, 80, 231, 20))
        self.ratesegHMM_radioButton.setChecked(True)
        self.ratesegHMM_radioButton.setAutoExclusive(True)
        self.ratesegHMM_radioButton.setObjectName("ratesegHMM_radioButton")
        self.ratesegHMM_radioButton.toggled.connect(self.model_radioButton_selected)
        ##Optional Version number choice group box
        self.version_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.version_groupBox.setGeometry(QtCore.QRect(180, 330, 111, 61))
        self.version_groupBox.setObjectName("version_groupBox")
        #Version 1 button
        self.v1_radioButton = QtWidgets.QRadioButton(self.version_groupBox)
        self.v1_radioButton.setGeometry(QtCore.QRect(10, 20, 100, 20))
        self.v1_radioButton.setObjectName("v1_radioButton")
        self.v1_radioButton.toggled.connect(self.version_radioButton_selected)
        #Version 2 button
        self.v2_radioButton = QtWidgets.QRadioButton(self.version_groupBox)
        self.v2_radioButton.setGeometry(QtCore.QRect(10, 40, 100, 20))
        self.v2_radioButton.setChecked(True)
        self.v2_radioButton.setObjectName("v2_radioButton")
        self.v2_radioButton.toggled.connect(self.version_radioButton_selected)
        ##K selection group box
        self.K_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.K_groupBox.setGeometry(QtCore.QRect(10, 330, 161, 80))
        self.K_groupBox.setObjectName("K_groupBox")
        #K label
        self.K_label = QtWidgets.QLabel(self.K_groupBox)
        self.K_label.setGeometry(QtCore.QRect(10, 20, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.K_label.setFont(font)
        self.K_label.setWordWrap(True)
        self.K_label.setObjectName("K_label")
        #K number spin box
        self.spinBox = QtWidgets.QSpinBox(self.K_groupBox)
        self.spinBox.setGeometry(QtCore.QRect(10, 50, 48, 24))
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(15)
        self.spinBox.setProperty("value", 5)
        self.spinBox.setObjectName("spinBox")
        self.spinBox.valueChanged.connect(self.k_selected)
        #Input text box for entry
        self.input_textbox = QtWidgets.QLineEdit(self.centralwidget)
        self.input_textbox.setGeometry(QtCore.QRect(10, 50, 231, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.input_textbox.setFont(font)
        self.input_textbox.setObjectName("input_textbox")
        
        self.browseButton = QtWidgets.QPushButton(self.centralwidget)
        self.browseButton.setGeometry(QtCore.QRect(250, 46, 100, 30))
        self.browseButton.setObjectName("browseButton")
        self.browseButton.clicked.connect(self.selectFile)
        
        #Status Bar     
        BeatMapSynth_GUI.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(BeatMapSynth_GUI)
        self.statusbar.setObjectName("statusbar")
        BeatMapSynth_GUI.setStatusBar(self.statusbar)
                
        self.retranslateUi(BeatMapSynth_GUI)
        QtCore.QMetaObject.connectSlotsByName(BeatMapSynth_GUI)

    def selectFile(self):
        from PyQt5.QtWidgets import QFileDialog
        self.input_textbox.setText((QFileDialog.getOpenFileName()[0]))
    
    def difficulty_radioButton_selected(self):
        if self.easy_radioButton.isChecked():
            self.difficulty = "easy"
        elif self.normal_radioButton.isChecked():
            self.difficulty = "normal"
        elif self.hard_radioButton.isChecked():
            self.difficulty = "hard"
        elif self.expert_radioButton.isChecked():
            self.difficulty = "expert"
        elif self.expertplus_radioButton.isChecked():
            self.difficulty = "expertPlus"

    def model_radioButton_selected(self):
        if self.random_radioButton.isChecked():
            self.model = "random"
        elif self.HMM_radioButton.isChecked():
            self.model = "HMM"
        elif self.segHMM_radioButton.isChecked():
            self.model = "segmented_HMM"
        elif self.ratesegHMM_radioButton.isChecked():
            self.model = "rate_modulated_segmented_HMM"

    def version_radioButton_selected(self):
        if self.v1_radioButton.isChecked():
            self.version = 1
        elif self.v2_radioButton.isChecked():
            self.version = 2
   
    def k_selected(self):
        self.k_value = self.spinBox.value()
    
    def execute(self):
        
        self.input_filepath = self.input_textbox.text()
        self.output_name = self.output_textbox.text()
        
        f = io.StringIO()
        with redirect_stdout(f):
            beatmapsynth.beat_map_synthesizer(self.input_filepath, self.output_name, self.difficulty, self.model, self.k_value, self.version)
        out = f.getvalue()
        self.statusbar.showMessage(out)        
    
    def retranslateUi(self, BeatMapSynth_GUI):
        _translate = QtCore.QCoreApplication.translate
        BeatMapSynth_GUI.setWindowTitle(_translate("BeatMapSynth_GUI", "BeatMapSynth"))
        self.input_header.setText(_translate("BeatMapSynth_GUI", "Music File"))
        self.output_header.setText(_translate("BeatMapSynth_GUI", "Output Name"))
        self.req_params_label.setText(_translate("BeatMapSynth_GUI", "Required Parameters"))
        self.opt_params_label.setText(_translate("BeatMapSynth_GUI", "Optional Parameters"))
        self.formats_label.setText(_translate("BeatMapSynth_GUI", "Accepted formats: .mp3, .wav, .flv, .raw, or .ogg"))
        self.output_label.setText(_translate("BeatMapSynth_GUI", "Text string that serves as the name of the exported zip folder and displayed name in Beat Saber"))
        self.output_textbox.setText(_translate("BeatMapSynth_GUI", "ex: \"Left Hand Free - Alt-J\""))
        self.browseButton.setText(_translate("BeatMapSynth_GUI", "Browse"))
        self.execute_button.setText(_translate("BeatMapSynth_GUI", "Create Song Mapping"))
        self.difficulty_groupBox.setTitle(_translate("BeatMapSynth_GUI", "Difficulty Level"))
        self.easy_radioButton.setText(_translate("BeatMapSynth_GUI", "Easy"))
        self.normal_radioButton.setText(_translate("BeatMapSynth_GUI", "Normal"))
        self.hard_radioButton.setText(_translate("BeatMapSynth_GUI", "Hard"))
        self.expert_radioButton.setText(_translate("BeatMapSynth_GUI", "Expert"))
        self.expertplus_radioButton.setText(_translate("BeatMapSynth_GUI", "Expert Plus"))
        self.model_groupBox.setTitle(_translate("BeatMapSynth_GUI", "Model"))
        self.random_radioButton.setText(_translate("BeatMapSynth_GUI", "Random"))
        self.HMM_radioButton.setText(_translate("BeatMapSynth_GUI", "HMM"))
        self.segHMM_radioButton.setText(_translate("BeatMapSynth_GUI", "Segmented HMM"))
        self.ratesegHMM_radioButton.setText(_translate("BeatMapSynth_GUI", "Rate Modulated/Segmented HMM"))
        self.version_groupBox.setTitle(_translate("BeatMapSynth_GUI", "Version"))
        self.v1_radioButton.setText(_translate("BeatMapSynth_GUI", "V1"))
        self.v2_radioButton.setText(_translate("BeatMapSynth_GUI", "V2 (default)"))
        self.K_groupBox.setTitle(_translate("BeatMapSynth_GUI", "K"))
        self.K_label.setText(_translate("BeatMapSynth_GUI", "Approx. number of segments for segmented models"))
        self.input_textbox.setText(_translate("BeatMapSynth_GUI", "ex: \"C:\\Desktop\\music\\My fave song.mp3\""))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    BeatMapSynth_GUI = QtWidgets.QMainWindow()
    ui = Ui_BeatMapSynth_GUI()
    ui.setupUi(BeatMapSynth_GUI)
    BeatMapSynth_GUI.show()
    sys.exit(app.exec_())
