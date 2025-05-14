import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import datetime
import os
from scipy.signal import find_peaks
import serial
import threading
import time
from pathlib import Path
import pandas as pd
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QSlider, QCheckBox, QFileDialog, QMessageBox,
    QGroupBox, QSizePolicy, QFrame, QStyleFactory
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

os.environ['QT_QUICK_CONTROLS_STYLE'] = 'Imagine'

class RecorderThread(QThread):
    status_update = pyqtSignal(str)
    recording_finished = pyqtSignal()
    data_received = pyqtSignal(list)
    buffer_processed = pyqtSignal()
    recording_started = pyqtSignal()  # New signal to indicate when recording actually starts

    def __init__(self, serial_port, baudrate):
        super().__init__()
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.recording = False
        self.serial_connection = None
        self.current_serial_line = ""
        self.data_buffer = []
        self.recording_started_flag = False
        self.min_buffer_size = 1

    def run(self):
        try:
            self.serial_connection = serial.Serial(self.serial_port, self.baudrate)
            buffer_count = 0
            while self.recording:
                if self.serial_connection and self.serial_connection.is_open and self.serial_connection.in_waiting:
                    # Read the data from serial port
                    data = self.serial_connection.readline().decode('utf-8').strip()
                    
                    # Check for initialization message
                    if "initok" in data:
                        self.status_update.emit("Initialisation capteur effectuée, prêt à recevoir des données...")
                        continue
                    
                    # Check for timeout message (end of recording session)
                    if "timeout" in data:
                        self.status_update.emit("Fin de capture détectée")
                        if self.data_buffer and len(self.data_buffer) >= self.min_buffer_size:
                            self.data_received.emit(self.data_buffer.copy())
                            # Reset for next recording
                            self.data_buffer = []
                            buffer_count = 0
                            self.recording_started_flag = False
                            self.status_update.emit("Attente de nouvelles données...")
                        continue
                    
                    # Process regular data
                    if data:
                        try:
                            # Clean and process the line
                            values = []
                            for val in data.split(','):
                                val = val.strip()
                                if val and val.isdigit():
                                    values.append(int(val))
                            
                            if values:
                                self.data_buffer.append(','.join(map(str, values)))
                                buffer_count += 1
                                
                                # If this is the first buffer with sufficient data, mark recording as started
                                if not self.recording_started_flag and len(values) > 10:
                                    self.recording_started_flag = True
                                    self.recording_started.emit()  # Emit signal when recording actually starts
                                    self.status_update.emit(f"Enregistrement démarré - Buffer reçu: {len(values)} valeurs")
                                else:
                                    self.status_update.emit(f"Buffer reçu: {len(values)} valeurs - {buffer_count} buffers traités - {buffer_count * 0.8:.2f} secondes d'enregistrement")
                                
                                # If we receive a complete buffer (200 values) followed by \r\n
                                if len(values) == 200:
                                    # Continue recording, this is a regular buffer
                                    pass
                        except ValueError as e:
                            self.status_update.emit(f"Erreur de conversion: {e}")
                            continue
                
                # Small pause to prevent CPU overload
                time.sleep(0.001)
                
        except Exception as e:
            self.status_update.emit(f"Erreur dans le thread d'enregistrement: {e}")
        finally:
            # Ensure serial port is closed if thread terminates
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
            self.recording_finished.emit()

    def stop(self):
        self.recording = False
        if self.data_buffer and len(self.data_buffer) >= self.min_buffer_size:
            self.data_received.emit(self.data_buffer.copy())


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi, constrained_layout=True)
        super().__init__(self.fig)


class AS7262Logger(QMainWindow):
    def __init__(self):
        super().__init__()

        # Variables pour la visualisation
        self.data = []
        self.AS7262_sample_rate = 250 # Hz provided by treadmill pulse High every N (=4) sample at 1000Hz
        self.window_size = 1000  # Nombre de points visibles par défaut
        self.time_step = 1 / self.AS7262_sample_rate
        self.max_y_value = 0
        self.show_peaks = False
        self.current_position = 0  # Position actuelle dans les données

        # Variables pour Gaitway
        self.gaitway_file_available = False
        self.gaitway_file_path = None
        self.show_gaitway_data = False
        self.lift_down_data = None
        self.lift_off_data = None
        self.target_speed_value = None

        # Variable pour les algorithmes Gaitway / Excel, Config par défaut :
        self.min_distance_period = 1.3  # Période entre pics en secondes
        self.search_window_size = 40  # Taille fenêtre recherche maximale
        self.ic_window_size = 0.1  # Fenêtre recherche IC en secondes
        self.tc_window_size = 0.8  # Fenêtre recherche TC en secondes

        # Variable for acceleration display
        self.show_acceleration = False
        # Variable for deceleration display
        self.show_deceleration = False
        # Variable for Algo IC display (nouvelle variable)
        self.show_algo_ic = False
        # Variable for Algo TC display
        self.show_algo_tc = False

        # Variables pour l'enregistrement
        self.serial_port = 'COM7'
        self.baudrate = 460800
        self.recording = False
        self.recorder_thread = None
        self.recording_status = "Arrêté"
        self.data_buffer = []
        self.start_timestamp = None  # Timestamp for when recording actually starts

        # Setup UI
        self.setWindowTitle("AS7262 Logger")
        self.setMinimumSize(1200, 800)
        self.setWindowIcon(QIcon("./logo.png"))

        self.setup_gui()

    def setup_gui(self):
        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # === SECTION ENREGISTREMENT ===
        recording_group = QGroupBox("Enregistrement")
        recording_layout = QHBoxLayout(recording_group)

        # Buttons
        self.rec_button = QPushButton("Rec")
        self.rec_button.setStyleSheet("background-color: #dc3545; color: white;")
        self.rec_button.clicked.connect(self.start_recording)
        recording_layout.addWidget(self.rec_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("background-color: #343a40; color: white;")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        recording_layout.addWidget(self.stop_button)

        # COM port
        recording_layout.addWidget(QLabel("Port COM:"))
        self.port_entry = QLineEdit(self.serial_port)
        self.port_entry.setFixedWidth(80)
        recording_layout.addWidget(self.port_entry)

        # Status
        recording_layout.addWidget(QLabel("État :"))
        self.status_label = QLabel(self.recording_status)
        recording_layout.addWidget(self.status_label)

        recording_layout.addStretch(1)
        main_layout.addWidget(recording_group)

        # === SECTION CONTENU PRINCIPAL ===
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # === SECTION VISUALISATION ===
        visualization_group = QGroupBox("Visualisation")
        visualization_layout = QVBoxLayout(visualization_group)

        # Top Frame (File selection)
        top_frame = QWidget()
        top_layout = QHBoxLayout(top_frame)
        top_layout.setContentsMargins(0, 0, 0, 0)

        self.select_button = QPushButton("Sélectionner un fichier log")
        self.select_button.setStyleSheet("background-color: #007bff; color: white;")
        self.select_button.clicked.connect(self.load_file)
        top_layout.addWidget(self.select_button)

        self.file_info = QLabel("Aucun fichier chargé")
        top_layout.addWidget(self.file_info)
        top_layout.addStretch(1)

        visualization_layout.addWidget(top_frame)

        # Scroll Frame
        scroll_frame = QWidget()
        scroll_layout = QVBoxLayout(scroll_frame)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        # Time slider
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(100)
        self.time_slider.valueChanged.connect(self.update_plot_from_slider)
        scroll_layout.addWidget(self.time_slider)

        # Navigation buttons
        nav_frame = QWidget()
        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(0, 0, 0, 0)

        prev_100_btn = QPushButton("<<")
        prev_100_btn.clicked.connect(lambda: self.navigate(-100))
        nav_layout.addWidget(prev_100_btn)

        prev_10_btn = QPushButton("<")
        prev_10_btn.clicked.connect(lambda: self.navigate(-10))
        nav_layout.addWidget(prev_10_btn)

        self.position_label = QLabel("Position: 0/0")
        nav_layout.addWidget(self.position_label)

        nav_layout.addStretch(1)

        next_10_btn = QPushButton(">")
        next_10_btn.clicked.connect(lambda: self.navigate(10))
        nav_layout.addWidget(next_10_btn)

        next_100_btn = QPushButton(">>")
        next_100_btn.clicked.connect(lambda: self.navigate(100))
        nav_layout.addWidget(next_100_btn)

        scroll_layout.addWidget(nav_frame)

        visualization_layout.addWidget(scroll_frame)

        # Graph Frame
        self.graph_frame = QWidget()
        graph_layout = QVBoxLayout(self.graph_frame)
        graph_layout.setContentsMargins(0, 0, 0, 0)

        # Create matplotlib canvas
        self.canvas = MatplotlibCanvas(self)
        graph_layout.addWidget(self.canvas)

        visualization_layout.addWidget(self.graph_frame, 1)  # Graph takes remaining space

        # === SECTION PARAMÈTRES (à droite) ===
        params_group = QGroupBox("Paramètres")
        params_layout = QVBoxLayout(params_group)
        params_group.setFixedWidth(130)

        # Window size label
        params_layout.addWidget(QLabel("Taille fenêtre"))

        # Window size slider
        self.window_slider = QSlider(Qt.Vertical)
        self.window_slider.setMinimum(50)
        self.window_slider.setMaximum(3000)
        self.window_slider.setValue(1000)
        self.window_slider.valueChanged.connect(self.update_window_size)
        params_layout.addWidget(self.window_slider, 1)

        # Peak detection checkbox
        self.peak_toggle = QCheckBox("Pic détection")
        self.peak_toggle.stateChanged.connect(self.toggle_peak_detection)
        params_layout.addWidget(self.peak_toggle)

        # Gaitway data checkbox
        self.gaitway_toggle = QCheckBox("IC/TC G3D")
        self.gaitway_toggle.stateChanged.connect(self.toggle_gaitway_data)
        params_layout.addWidget(self.gaitway_toggle)

        self.accelerattion_graph = QCheckBox("Accélération")
        self.accelerattion_graph.stateChanged.connect(self.toogle_acceleration_signal)
        params_layout.addWidget(self.accelerattion_graph)

        self.deceleration_graph = QCheckBox("Décélération")
        self.deceleration_graph.stateChanged.connect(self.toggle_deceleration_signal)
        params_layout.addWidget(self.deceleration_graph)

        # Ajout du nouveau bouton "Algo IC"
        self.algo_ic_graph = QCheckBox("Algo IC")
        self.algo_ic_graph.stateChanged.connect(self.toggle_algo_ic)
        params_layout.addWidget(self.algo_ic_graph)

        # Ajout du nouveau bouton "Algo TC"
        self.algo_tc_graph = QCheckBox("Algo TC")
        self.algo_tc_graph.stateChanged.connect(self.toggle_algo_tc)
        params_layout.addWidget(self.algo_tc_graph)

        # Bouton d'export CSV
        self.export_button = QPushButton("Export to CSV")
        self.export_button.setStyleSheet("background-color: #28a745; color: white;")
        self.export_button.clicked.connect(self.export_to_csv)
        params_layout.addWidget(self.export_button)

        # Add visualization and params sections to content
        content_layout.addWidget(visualization_group, 1)
        content_layout.addWidget(params_group)

        main_layout.addWidget(content_widget, 1)

    def set_detection_parameters(self):
        """Configure les paramètres de détection selon la vitesse cible"""
        if self.target_speed_value is None:
            return

        speed_kmh = self.target_speed_value * 0.36

        # Arrondir pour éviter les erreurs de comparaison flottante
        rounded_speed = round(speed_kmh, 1)

        if abs(rounded_speed - 16.1) < 0.2:  # Vitesse lente (environ 3 km/h)
            self.min_distance_period = 0.64
            self.search_window_size = 20
            self.ic_window_size = 0.04
            self.tc_window_size = 0.2
        elif abs(rounded_speed - 3.0) < 0.2:  # Vitesse rapide (environ 16 km/h)
            self.min_distance_period = 1.3
            self.search_window_size = 35
            self.ic_window_size = 0.1
            self.tc_window_size = 0.8
        else:
            # Message d'avertissement si vitesse non reconnue
            self.update_status(f"Attention: Variables de détection automatique non définies pour {rounded_speed} km/h")
            QMessageBox.warning(self, "Attention",
                                f"Variables de détection automatique non définies pour {rounded_speed} km/h.\n"
                                "Utilisation des paramètres par défaut.")

    def navigate(self, step):
        """Naviguer dans les données par un nombre de pas donner"""
        if not self.data:
            return

        max_position = max(0, len(self.data) - self.window_size)
        new_position = min(max(0, self.current_position + step), max_position)

        self.current_position = new_position
        self.time_slider.setValue(new_position)
        self.update_plot(new_position)

    def check_for_gaitway_file(self, filename):
            """Vérifie si un fichier Gaitway correspondant existe"""
            base_name = os.path.basename(filename)
            base_path = os.path.dirname(filename)

            # Créer le nom du fichier Gaitway correspondant
            if base_name.startswith("log_") and base_name.endswith(".txt"):
                # Extraire la partie timestamp du nom
                timestamp_part = base_name[4:-4]  # Enlève "log_" et ".txt"
                gaitway_filename = f"log_{timestamp_part}_G.txt"
                gaitway_filepath = os.path.join(base_path, gaitway_filename)

                # Vérifier si le fichier existe
                if os.path.exists(gaitway_filepath):
                    self.gaitway_file_available = True
                    self.gaitway_file_path = gaitway_filepath

                    # Lire la vitesse cible du fichier Gaitway
                    try:
                        data = pd.read_csv(gaitway_filepath, skiprows=44, delimiter='\t', usecols=[20], nrows=100,
                                           low_memory=False)
                        if len(data) > 50:
                            self.target_speed_value = data.iloc[50, 0] * 10
                            # Configurer les paramètres selon la vitesse
                            self.set_detection_parameters()
                        else:
                            self.target_speed_value = None
                    except Exception as e:
                        print(f"Erreur lecture vitesse: {e}")
                        self.target_speed_value = None

                    return True

            self.gaitway_file_available = False
            self.gaitway_file_path = None
            self.target_speed_value = None
            return False

    def read_gaitway_data(self):
        """Lit et traite les données du fichier Gaitway"""
        if not self.gaitway_file_available or not self.gaitway_file_path:
            return False

        try:
            # Utiliser la fonction fournie pour lire et extraire les données
            self.lift_off_data, self.lift_down_data = self.read_and_extract(self.gaitway_file_path)
            return True
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier Gaitway: {e}")
            QMessageBox.critical(self, "Erreur", f"Impossible de lire le fichier Gaitway: {e}")
            self.gaitway_file_available = False
            return False

    def read_and_extract(self, file_path):
        """Lit et extrait les données du fichier Gaitway"""
        data = pd.read_csv(file_path, skiprows=44, delimiter='\t', usecols=[0, 20, 30, 31], low_memory=False)
        data.columns = ['Time', 'Target_Speed', 'Contact_Side', 'Contact_Mode']
        time = data['Time'].values
        contact_side = data['Contact_Side']
        contact_mode = data['Contact_Mode']
        target_speed = data['Target_Speed']

        lift_down = []
        lift_off = []

        # Déterminer si la vitesse cible est > 25 m/s
        is_high_speed = False
        try:
            # Essayer de lire la 50e valeur pour déterminer la vitesse
            if len(data) > 50 and target_speed is not None:
                target_speed_value = target_speed.iloc[50] * 10
                is_high_speed = target_speed_value > 24
        except:
            is_high_speed = False

        if contact_side is not None and contact_mode is not None:
            if not is_high_speed:  # Vitesse < 25 m/s
                # Détecter les lift_down (pose)
                previous_side = None
                for i in range(len(contact_side)):
                    if pd.notna(contact_side.iloc[i]):
                        current_side = contact_side.iloc[i]
                        if (previous_side is None or previous_side == 'Left') and current_side == 'Right':
                            lift_down.append((time[i], 'Pose du Pied'))
                        previous_side = current_side

                # Détecter les lift_off (levé)
                previous_mode = None
                for i in range(len(contact_side)):
                    if pd.notna(contact_side.iloc[i]) and pd.notna(contact_mode.iloc[i]):
                        current_side = contact_side.iloc[i]
                        current_mode = contact_mode.iloc[i]
                        if (current_side == 'Left' and (previous_mode is None or previous_mode == 'DC')
                                and current_mode == 'SC'):
                            lift_off.append((time[i], 'levé de Pied'))
                        previous_mode = current_mode
            else:  # Vitesse > 25 m/s
                # Détecter les lift_down
                previous_side = None
                for i in range(len(contact_side)):
                    if pd.notna(contact_side.iloc[i]):
                        current_side = contact_side.iloc[i]
                        if (previous_side is None or previous_side == 'Left') and current_side == 'Right':
                            lift_down.append((time[i], 'Pose du Pied'))
                        previous_side = current_side

                # Détecter les lift_off
                previous_mode = None
                for i in range(len(contact_mode)):
                    if pd.notna(contact_mode.iloc[i]) and pd.notna(contact_side.iloc[i]):
                        current_mode = contact_mode.iloc[i]
                        current_side = contact_side.iloc[i]
                        if current_side == 'Right' and (
                                previous_mode is None or previous_mode == 'SC') and current_mode == 'Aerial':
                            lift_off.append((time[i], 'levé de Pied'))
                        previous_mode = current_mode

        # Créer des DataFrames
        df_lift_off = pd.DataFrame(lift_off, columns=['Temps (s)', 'Type'])
        df_lift_down = pd.DataFrame(lift_down, columns=['Temps (s)', 'Type'])

        return df_lift_off, df_lift_down

    def toggle_gaitway_data(self):
        """Active/désactive l'affichage des données Gaitway"""
        self.show_gaitway_data = self.gaitway_toggle.isChecked()

        if self.gaitway_file_available and self.show_gaitway_data:
            # Si on active l'affichage et qu'on n'a pas encore lu les données
            if self.lift_off_data is None or self.lift_down_data is None:
                success = self.read_gaitway_data()
                if not success:
                    self.show_gaitway_data = False
                    self.gaitway_toggle.setChecked(False)
                    return
        elif self.show_gaitway_data and not self.gaitway_file_available:
            # Si aucun fichier Gaitway n'est disponible, informer l'utilisateur
            QMessageBox.information(self, "Information", "Aucun fichier Gaitway disponible pour ce log.")
            self.gaitway_toggle.setChecked(False)
            self.show_gaitway_data = False
            return

        self.update_plot()

    def toogle_acceleration_signal(self):
        """Active ou désactive le tracé d'accélération"""
        self.show_acceleration = self.accelerattion_graph.isChecked()
        self.update_plot()

    def toggle_deceleration_signal(self):
        """Active ou désactive le tracé de décélération"""
        self.show_deceleration = self.deceleration_graph.isChecked()
        self.update_plot()

    def toggle_algo_ic(self):
        """Active ou désactive la détection des poses par l'algorithme IC"""
        self.show_algo_ic = self.algo_ic_graph.isChecked()
        self.update_plot()

    def toggle_algo_tc(self):
        """Active ou désactive la détection des fin de contact par l'algorithme TC"""
        self.show_algo_tc = self.algo_tc_graph.isChecked()
        self.update_plot()

    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner un fichier log",
            "",
            "Fichiers texte (*.txt);;Tous les fichiers (*.*)"
        )

        if filename:
            # Extraction de la date et de l'heure depuis le nom du fichier
            basename = os.path.basename(filename)
            date_display = "Date inconnue"
            if basename.startswith("log_") and basename.endswith(".txt"):
                date_str = basename[4:-4]  # Enlève "log_" et ".txt"
                try:
                    file_date = datetime.datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                    date_display = file_date.strftime("%d/%m/%Y à %H:%M")
                except ValueError:
                    pass

            # Lecture et traitement du fichier
            self.data = []
            try:
                with open(filename, 'r') as f:
                    for line in f:
                        try:
                            values = [int(x) for x in line.strip().split(',') if x]
                            self.data.extend(values)
                        except ValueError:
                            # Ignorer les lignes qui ne contiennent pas de nombres
                            continue
            except Exception as e:
                self.file_info.setText(f"Erreur: {str(e)}")
                return

            # Calcul de la durée
            duration = len(self.data) / self.AS7262_sample_rate
            duration_str = f"{duration:.1f}"

            # Mise à jour du max_y_value
            if self.data:
                self.max_y_value = max(self.data) + 3
            else:
                self.max_y_value = 100  # Valeur par défaut si pas de données

            # Vérifier si un fichier Gaitway correspondant existe
            gaitway_available = self.check_for_gaitway_file(filename)

            # Mise à jour de l'information du fichier
            file_info_text = f"Fichier sélectionné :  Date : {date_display}    /    Durée totale : {duration_str} secondes"
            if gaitway_available:
                file_info_text += "   /   Fichier G3D disponible "
                if self.target_speed_value is not None:
                    speed_kmh = self.target_speed_value * 0.36
                    file_info_text += f"   /   {speed_kmh:.1f} km/h"
            self.file_info.setText(file_info_text)

            # Réinitialiser les variables de Gaitway
            self.show_gaitway_data = False
            self.lift_off_data = None
            self.lift_down_data = None
            self.show_acceleration = False  # Réinitialiser l'affichage de l'accélération
            self.show_deceleration = False  # Réinitialiser l'affichage de la décélération
            self.show_algo_ic = False  # Réinitialiser l'affichage de Algo IC
            self.show_algo_tc = False  # Réinitialiser l'affichage de Algo TC

            # Réinitialiser le toggle Gaitway si actif
            if self.gaitway_toggle.isChecked():
                self.gaitway_toggle.setChecked(False)

            # Réinitialiser le toggle d'accélération si actif
            if self.accelerattion_graph.isChecked():
                self.accelerattion_graph.setChecked(False)
                
            # Réinitialiser le toggle de décélération si actif
            if self.deceleration_graph.isChecked():
                self.deceleration_graph.setChecked(False)
                
            # Réinitialiser le toggle de Algo IC si actif
            if self.algo_ic_graph.isChecked():
                self.algo_ic_graph.setChecked(False)
                
            # Réinitialiser le toggle de Algo TC si actif
            if self.algo_tc_graph.isChecked():
                self.algo_tc_graph.setChecked(False)

            # Réinitialiser la position
            self.current_position = 0

            # Mise à jour du slider
            max_position = max(0, len(self.data) - self.window_size)
            self.time_slider.setMaximum(max_position)
            self.time_slider.setValue(0)

            # Affichage initial
            self.update_plot()

    def update_window_size(self):
        """Met à jour la taille de la fenêtre d'affichage"""
        self.window_size = self.window_slider.value()
        if self.data:
            # Mettre à jour le slider après changement de taille
            max_position = max(0, len(self.data) - self.window_size)
            self.time_slider.setMaximum(max_position)

            # Ajuster la position actuelle si nécessaire
            if self.current_position > max_position:
                self.current_position = max_position
                self.time_slider.setValue(self.current_position)

            self.update_plot()

    def update_plot_from_slider(self, value):
        """Met à jour le graphique basé sur la position du curseur"""
        if not self.data:
            return

        position = int(value)
        self.current_position = position
        self.update_plot(position)

    def update_plot(self, start_idx=None):
        """Met à jour le graphique avec les données actuelles"""
        if not self.data:
            return

        if start_idx is None:
            start_idx = self.current_position

        end_idx = start_idx + self.window_size

        # Extraction des données à afficher
        display_data = self.data[start_idx:min(end_idx, len(self.data))]

        # Calcul de l'axe temporel
        time_axis = np.arange(len(display_data)) * self.time_step + (start_idx * self.time_step)

        # Mise à jour du graphique
        self.canvas.ax.clear()
        self.canvas.ax.plot(time_axis, display_data, label='Signal')

        peaks = []  # Variable pour stocker les indices des pics détectés

        # Si la détection de pics est activée
        if self.show_peaks and len(display_data) > 0:
            try:
                # Conversion de temps en secondes (640ms) en nombre de points
                min_distance = int(self.min_distance_period * self.AS7262_sample_rate)

                # Utilisation de find_peaks avec des paramètres adaptés pour détecter les pics
                peaks, properties = find_peaks(
                    display_data, 
                    distance=min_distance,    # Distance minimale entre pics
                    prominence=0.3,           # Prominence pour filtrer les pics significatifs : 0.3
                    height=0                  # Seulement des pics positifs
                )
                
                # Pour chaque pic détecté, trouver le premier instant de la valeur maximale
                first_max_indices = []
                for peak_idx in peaks:
                    # Obtenir la valeur du pic
                    peak_value = display_data[peak_idx]
                    
                    # Rechercher en arrière pour trouver le premier point où cette valeur apparaît
                    # dans une fenêtre qui s'étend avant le pic détecté
                    search_window = self.search_window_size
                    start_search = max(0, peak_idx - search_window)
                    
                    # Trouver le premier indice où la valeur est égale au pic
                    for i in range(start_search, peak_idx + 1):
                        if display_data[i] == peak_value:
                            first_max_indices.append(i)
                            break
                
                # Tracer les pics détectés au premier instant de valeur maximale
                if first_max_indices:
                    self.canvas.ax.plot(time_axis[first_max_indices], np.array(display_data)[first_max_indices], "ro", label='Pics détectés')
                    
                    # Utilisez first_max_indices comme nouvelle variable peaks pour la détection Algo IC
                    peaks = first_max_indices
                    
                    # Ajouter une info sur la fréquence des pics détectés
                    if len(first_max_indices) > 1:
                        avg_period = np.mean(np.diff(first_max_indices)) * self.time_step
                        self.canvas.ax.set_title(f'Visualisation des données - Période moyenne des pics: {avg_period:.3f}s')
                        
            except Exception as e:
                print(f"Erreur lors de la détection des pics: {e}")

        # Si l'affichage des données Gaitway est activé
        if self.show_gaitway_data and self.gaitway_file_available and self.lift_off_data is not None and self.lift_down_data is not None:
            # Calculer les limites temporelles de l'affichage actuel
            start_time = start_idx * self.time_step
            end_time = (start_idx + len(display_data)) * self.time_step

            # Pour la légende, nous devons ajouter des lignes fictives
            self.canvas.ax.plot([], [], color='blue', linestyle='-', label='Contact Initial (G3D)')
            self.canvas.ax.plot([], [], color='yellow', linestyle='-', label='Fin de Contact (G3D)')

            # Filtrer les données Gaitway pour n'afficher que celles dans la fenêtre temporelle
            if 'Temps (s)' in self.lift_down_data:
                lift_down_times = self.lift_down_data['Temps (s)']
                # Afficher les lift_down (poses) comme des barres verticales bleues
                for time_val in lift_down_times:
                    if start_time <= time_val <= end_time:
                        self.canvas.ax.axvline(x=time_val, color='blue', linestyle='-', alpha=0.7)

            if 'Temps (s)' in self.lift_off_data:
                lift_off_times = self.lift_off_data['Temps (s)']
                # Afficher les lift_off (levés) comme des barres verticales jaunes
                for time_val in lift_off_times:
                    if start_time <= time_val <= end_time:
                        self.canvas.ax.axvline(x=time_val, color='yellow', linestyle='-', alpha=0.7)

        # Calcul des dérivées pour l'accélération et la décélération
        derivatives = None
        if len(display_data) > 1:
            # Calculer la dérivée du signal une seule fois pour toutes les fonctionnalités
            derivatives = np.diff(display_data) * self.AS7262_sample_rate
            # Créer un nouvel axe de temps pour les dérivées (1 point de moins)
            der_time_axis = time_axis[:-1]
            
            # Si l'affichage de l'accélération est activé
            if self.show_acceleration:
                # Seulement les valeurs positives (accélération)
                acceleration = np.maximum(0, derivatives)
                # Tracer l'accélération en rouge
                self.canvas.ax.plot(der_time_axis, acceleration, 'r-', label='Accélération')
                
            # Si l'affichage de la décélération est activé
            if self.show_deceleration:
                # Seulement les valeurs négatives (décélération), transformées en positif pour l'affichage
                deceleration = -np.minimum(0, derivatives)
                # Tracer la décélération en vert
                self.canvas.ax.plot(der_time_axis, deceleration, 'g-', label='Décélération')

        # Implémenter l'algorithme IC (détection de la première accélération avant chaque pic)
        if self.show_algo_ic and self.show_peaks and len(peaks) > 0 and derivatives is not None:
            # Calculer la fenêtre de recherche en nombre de points (0.04s)
            window_size_points = int(self.ic_window_size * self.AS7262_sample_rate)
            
            # Points détectés par l'algorithme IC
            ic_points_x = []
            ic_points_y = []
            
            for peak_idx in peaks:
                # Calculer l'indice de début de la fenêtre de recherche
                start_search_idx = max(0, peak_idx - window_size_points)
                
                # Extraire les dérivées dans la fenêtre d'intérêt
                # Note: derivatives a un indice de moins que display_data
                search_window = derivatives[start_search_idx:peak_idx] if peak_idx > 0 else []
                
                # Chercher la première accélération (dérivée positive) dans la fenêtre
                found_acceleration = False
                for i, derivative in enumerate(search_window):
                    if derivative > 0:  # C'est une accélération
                        # Calculer l'indice exact dans le signal original
                        acceleration_idx = start_search_idx + i
                        # Ajouter ce point aux listes pour l'affichage
                        ic_points_x.append(time_axis[acceleration_idx])
                        ic_points_y.append(display_data[acceleration_idx])
                        found_acceleration = True
                        break
            
            # Afficher les points détectés par Algo IC
            if ic_points_x:
                self.canvas.ax.scatter(ic_points_x, ic_points_y, color='blue', marker='o', 
                                      label='Algo IC', s=50, zorder=5)
            
        # Implémenter l'algorithme TC (détection de la dernière décélération après chaque pic)
        if self.show_algo_tc and self.show_peaks and len(peaks) > 0 and derivatives is not None:
            # Calculer la fenêtre de recherche en nombre de points
            window_size_points = int(self.tc_window_size * self.AS7262_sample_rate)
            
            # Points détectés par l'algorithme TC
            tc_points_x = []
            tc_points_y = []
            
            for peak_idx in peaks:
                # Calculer l'indice de fin de la fenêtre de recherche
                end_search_idx = min(len(display_data) - 1, peak_idx + window_size_points)
                
                # Extraire les dérivées dans la fenêtre d'intérêt
                # Note: derivatives a un indice de moins que display_data
                search_window = derivatives[peak_idx:end_search_idx] if peak_idx < len(derivatives) else []
                
                # Chercher la dernière décélération (dérivée négative) dans la fenêtre
                last_deceleration_idx = None
                for i in range(len(search_window)):
                    actual_idx = peak_idx + i
                    if actual_idx < len(derivatives) and derivatives[actual_idx] < 0:  # C'est une décélération
                        # Mémoriser cet indice comme candidat
                        last_deceleration_idx = actual_idx
                
                # Si on a trouvé une décélération, l'ajouter aux points à afficher
                if last_deceleration_idx is not None:
                    tc_points_x.append(time_axis[last_deceleration_idx])
                    tc_points_y.append(display_data[last_deceleration_idx])
            
            # Afficher les points détectés par Algo TC
            if tc_points_x:
                self.canvas.ax.scatter(tc_points_x, tc_points_y, color='yellow', marker='o', 
                                      label='Algo TC', s=50, zorder=5)
            
        # Mise à jour du label de position
        position_info = f"Position: {start_idx}/{len(self.data)} - Fenêtre: {self.window_size} points"
        self.position_label.setText(position_info)

        # Information axes
        self.canvas.ax.set_xlabel(f'Temps (s)')
        self.canvas.ax.set_ylabel('Intensité lumineuse')
        self.canvas.ax.grid(True)
        self.canvas.ax.set_title(f'Visualisation des données')

        # Fixation de l'échelle y
        self.canvas.ax.set_ylim(0, self.max_y_value)
        
        # Afficher la légende si nécessaire
        if (self.show_acceleration or self.show_deceleration or self.show_algo_ic or 
            self.show_algo_tc or self.show_peaks or 
            (self.show_gaitway_data and self.gaitway_file_available and 
             self.lift_off_data is not None and self.lift_down_data is not None)):
            self.canvas.ax.legend()

        # Rafraîchissement du canvas
        self.canvas.draw()

    def toggle_peak_detection(self):
        """Active/désactive la détection des pics"""
        self.show_peaks = self.peak_toggle.isChecked()
        self.update_plot()

    def start_recording(self):
        """Démarre l'enregistrement des données depuis le port série"""
        if self.recording:
            return  # Déjà en cours d'enregistrement

        # Récupérer le port COM depuis l'interface
        self.serial_port = self.port_entry.text()

        try:
            # Créer et démarrer le thread d'enregistrement
            self.recorder_thread = RecorderThread(self.serial_port, self.baudrate)
            self.recorder_thread.status_update.connect(self.update_status)
            self.recorder_thread.data_received.connect(self.save_data)
            self.recorder_thread.recording_finished.connect(self.handle_recording_finished)
            self.recorder_thread.recording_started.connect(self.handle_recording_started)

            # Marquer comme en cours d'enregistrement et démarrer
            self.recording = True
            self.recorder_thread.recording = True
            self.recorder_thread.start()

            # Mettre à jour l'interface
            self.recording_status = "Initialisation et purge du buffer..."
            self.status_label.setText(self.recording_status)
            self.rec_button.setEnabled(False)
            self.stop_button.setEnabled(True)

        except Exception as e:
            self.recording_status = f"Erreur: {str(e)}"
            self.status_label.setText(self.recording_status)
            print(f"Erreur de démarrage: {e}")

    def handle_recording_started(self):
        """Appelé quand l'enregistrement commence réellement"""
        self.start_timestamp = datetime.datetime.now()
        print(f"Enregistrement démarré à: {self.start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    def update_status(self, message):
        """Met à jour le statut d'enregistrement dans l'interface"""
        self.recording_status = message
        self.status_label.setText(self.recording_status)
        print(message)  # Afficher aussi dans la console

    def save_data(self, data_buffer):
        """Sauvegarde les données enregistrées dans un fichier"""
        if not data_buffer:
            self.update_status("Pas assez de données pour sauvegarder un fichier")
            return

        # Utilisation du timestamp de début d'enregistrement pour le nom de fichier
        if self.start_timestamp:
            timestamp = self.start_timestamp.strftime("%Y%m%d_%H%M%S")
        else:
            # Fallback au timestamp actuel si pour une raison quelconque le timestamp de départ n'est pas disponible
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
        filename = f"log_{timestamp}.txt"

        # Création du dossier logs s'il n'existe pas
        Path("logs").mkdir(exist_ok=True)

        # Sauvegarde des données
        filepath = Path("logs") / filename
        try:
            with open(filepath, 'w') as f:
                for line in data_buffer:
                    f.write(line + '\n')

            self.update_status(f"Données sauvegardées dans: {filepath}")

            # Compter le nombre total d'échantillons
            total_samples = 0
            for line in data_buffer:
                total_samples += len(line.split(','))

            # Proposer de charger automatiquement ce fichier seulement si suffisamment de données
            if total_samples > 100:  # Seuil arbitraire pour éviter les fichiers trop petits
                self.ask_load_saved_file(str(filepath))
            else:
                self.update_status(
                    f"Fichier sauvegardé avec {total_samples} échantillons (trop petit pour être chargé automatiquement)")

        except Exception as e:
            self.update_status(f"Erreur lors de la sauvegarde: {e}")

    def ask_load_saved_file(self, filepath):
        """Demande à l'utilisateur s'il veut charger le fichier qui vient d'être sauvegardé"""
        answer = QMessageBox.question(
            self,
            "Fichier sauvegardé",
            f"Le fichier a été sauvegardé dans {filepath}.\nVoulez-vous le charger maintenant?",
            QMessageBox.Yes | QMessageBox.No
        )

        if answer == QMessageBox.Yes:
            self.load_saved_file(filepath)

    def load_saved_file(self, filepath):
        """Charge le fichier qui vient d'être sauvegardé"""
        # Lecture et traitement du fichier
        self.data = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        values = [int(x) for x in line.strip().split(',') if x]
                        self.data.extend(values)
                    except ValueError:
                        # Ignorer les lignes qui ne contiennent pas de nombres
                        continue

            # Extraire les informations du fichier
            basename = os.path.basename(filepath)
            date_display = "Date inconnue"

            if basename.startswith("log_") and basename.endswith(".txt"):
                date_str = basename[4:-4]  # Enlève "log_" et ".txt"
                try:
                    file_date = datetime.datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                    date_display = file_date.strftime("%d/%m/%Y à %H:%M")
                except ValueError:
                    pass

            # Calcul de la durée
            duration = len(self.data) / self.AS7262_sample_rate
            duration_str = f"{duration:.1f}"

            # Mise à jour du max_y_value
            if self.data:
                self.max_y_value = max(self.data) + 3
            else:
                self.max_y_value = 100

            # Mise à jour de l'information du fichier
            self.file_info.setText(
                f"Fichier sélectionné : Date : {date_display}    /    Durée totale : {duration_str} secondes"
            )

            # Réinitialiser la position
            self.current_position = 0

            # Mise à jour du slider
            max_position = max(0, len(self.data) - self.window_size)
            self.time_slider.setMaximum(max_position)
            self.time_slider.setValue(0)

            # Affichage initial
            self.update_plot()

        except Exception as e:
            self.file_info.setText(f"Erreur lors du chargement: {str(e)}")

    def stop_recording(self):
        """Arrête l'enregistrement des données"""
        if not self.recording:
            return

        # Marquer l'arrêt de l'enregistrement
        self.recording = False

        # Arrêter le thread d'enregistrement
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop()
            self.recorder_thread.wait(1000)  # Attendre max 1 seconde que le thread se termine

        # Mettre à jour l'interface
        self.recording_status = "Arrêté"
        self.status_label.setText(self.recording_status)
        self.rec_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def handle_recording_finished(self):
        """Appelé quand le thread d'enregistrement se termine"""
        self.recording = False
        self.recording_status = "Arrêté"
        self.status_label.setText(self.recording_status)
        self.rec_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def export_to_csv(self):
        """Exporte les données du graphique actuel vers un fichier CSV"""
        if not self.data:
            QMessageBox.warning(self, "Attention", "Aucune donnée à exporter")
            return

        # Demande à l'utilisateur où sauvegarder le fichier CSV
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Exporter en CSV", "", "Fichiers CSV (*.csv);;Tous les fichiers (*.*)", 
            options=options
        )

        if not filename:
            return  # L'utilisateur a annulé

        if not filename.endswith('.csv'):
            filename += '.csv'

        try:
            with open(filename, 'w', newline='') as csvfile:
                # Préparation des données pour l'export
                
                # 1. Information sur le fichier (première ligne)
                file_info = {}
                basename = os.path.basename(filename)
                date_display = "Date inconnue"
                if basename.startswith("log_") and basename.endswith(".txt"):
                    date_str = basename[4:-4]  # Enlève "log_" et ".txt"
                    try:
                        file_date = datetime.datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                        date_display = file_date.strftime("%d/%m/%Y à %H:%M")
                    except ValueError:
                        pass
                duration_str = f"{len(self.data) / self.AS7262_sample_rate:.1f}"
                gaitway_available = "Oui" if self.gaitway_file_available else "Non"
                
                if hasattr(self, 'file_info') and self.file_info.text() != "Aucun fichier chargé":
                    # Extraire le nom du fichier d'origine
                    file_info_text = self.file_info.text()
                    basename = file_info_text.split(":")[1].strip() if ":" in file_info_text else basename

                header_info = {
                    "Nom du fichier": basename,
                    "Date": date_display,
                    "Durée": duration_str,
                    "Fichier Gaitway disponible": gaitway_available
                }
                
                # 2. Préparation des colonnes de données
                data_dict = {}
                
                # Colonne temps
                time_values = np.arange(len(self.data)) * self.time_step
                data_dict["Temps (s)"] = time_values
                
                # Colonne intensité lumineuse
                data_dict["Intensité lumineuse"] = self.data
                
                # Colonne détection de pics
                peaks_column = np.zeros(len(self.data))
                try:
                    # Utiliser les mêmes paramètres que dans la visualisation
                    min_distance = int(self.min_distance_period * self.AS7262_sample_rate)
                    
                    # Détection des pics sur l'ensemble des données
                    peaks, _ = find_peaks(
                        self.data, 
                        distance=min_distance,
                        prominence=0.3,
                        height=0
                    )
                    
                    # Pour chaque pic, trouver le premier instant de valeur maximale
                    first_max_indices = []
                    for peak_idx in peaks:
                        # Obtenir la valeur du pic
                        peak_value = self.data[peak_idx]

                        # Rechercher en arrière pour trouver le premier point où cette valeur apparaît
                        search_window = self.search_window_size
                        start_search = max(0, peak_idx - search_window)
                        
                        for i in range(start_search, peak_idx + 1):
                            if self.data[i] == peak_value:
                                first_max_indices.append(i)
                                break
                    
                    # Marquer les positions des pics détectés
                    peaks_column[first_max_indices] = 1
                    
                    # Ajouter une information sur les intervalles moyens entre pics
                    if len(first_max_indices) > 1:
                        avg_period = np.mean(np.diff(first_max_indices)) * self.time_step
                        header_info["Période moyenne des pics"] = f"{avg_period:.3f} s"
                        header_info["Nombre de pics détectés"] = str(len(first_max_indices))
                except Exception as e:
                    print(f"Erreur lors de la détection des pics pour l'export CSV: {e}")
                
                # Ajouter la colonne de pics détectés
                data_dict["Pics (1=oui, 0=non)"] = peaks_column
                
                # Colonne Pose/Levé (si activé)
                if self.show_gaitway_data and self.gaitway_file_available and self.lift_off_data is not None and self.lift_down_data is not None:
                    pose_leve = np.zeros(len(self.data))
                    
                    # Marquage des poses (lift_down)
                    if 'Temps (s)' in self.lift_down_data:
                        for time_val in self.lift_down_data['Temps (s)']:
                            # Trouver l'index temporel le plus proche
                            idx = int(time_val / self.time_step)
                            if 0 <= idx < len(pose_leve):
                                pose_leve[idx] = 2
                    
                    # Marquage des levés (lift_off)
                    if 'Temps (s)' in self.lift_off_data:
                        for time_val in self.lift_off_data['Temps (s)']:
                            # Trouver l'index temporel le plus proche
                            idx = int(time_val / self.time_step)
                            if 0 <= idx < len(pose_leve):
                                pose_leve[idx] = 1
                    
                    data_dict["Pose_Leve (2=pose, 1=leve)"] = pose_leve
                
                # Colonnes accélération/décélération (si activées)
                if self.show_acceleration or self.show_deceleration or self.show_algo_ic or self.show_algo_tc:
                    # Calculer les dérivées
                    derivatives = np.diff(self.data) * self.AS7262_sample_rate
                    # Ajouter un zéro pour avoir la même longueur
                    derivatives = np.append(derivatives, 0)
                    
                    if self.show_acceleration:
                        # Marquer 1 quand il y a accélération (dérivée positive)
                        acceleration = np.zeros(len(self.data))
                        acceleration[:-1] = np.where(derivatives[:-1] > 0, 1, 0)
                        data_dict["Accélération (1=oui)"] = acceleration
                    
                    if self.show_deceleration:
                        # Marquer 1 quand il y a décélération (dérivée négative)
                        deceleration = np.zeros(len(self.data))
                        deceleration[:-1] = np.where(derivatives[:-1] < 0, 1, 0)
                        data_dict["Décélération (1=oui)"] = deceleration
                    
                    # Ajouter les détections Algo IC si activées
                    if self.show_algo_ic and self.show_peaks:
                        # Détection des pics sur l'ensemble des données
                        try:
                            min_distance = int(self.min_distance_period * self.AS7262_sample_rate)
                            peaks, _ = find_peaks(
                                self.data, 
                                distance=min_distance,
                                prominence=0.3,
                                height=0
                            )
                            
                            # Pour chaque pic, trouver le premier instant de la valeur maximale
                            first_max_indices = []
                            for peak_idx in peaks:
                                peak_value = self.data[peak_idx]
                                search_window = self.search_window_size
                                start_search = max(0, peak_idx - search_window)
                                
                                for i in range(start_search, peak_idx + 1):
                                    if self.data[i] == peak_value:
                                        first_max_indices.append(i)
                                        break
                            
                            # Créer une colonne pour les points Algo IC
                            algo_ic_column = np.zeros(len(self.data))
                            
                            # Pour chaque pic, chercher la première accélération dans la fenêtre
                            window_size_points = int(self.ic_window_size * self.AS7262_sample_rate)
                            
                            for peak_idx in first_max_indices:
                                start_search_idx = max(0, peak_idx - window_size_points)
                                
                                # Chercher la première accélération
                                for i in range(start_search_idx, peak_idx):
                                    if i < len(derivatives) - 1 and derivatives[i] > 0:
                                        algo_ic_column[i] = 1
                                        break
                            
                            data_dict["Algo_IC (1=détecté)"] = algo_ic_column
                        except Exception as e:
                            print(f"Erreur lors de la détection Algo IC pour l'export CSV: {e}")
                    
                    # Ajouter les détections Algo TC si activées
                    if self.show_algo_tc and self.show_peaks:
                        # Détection des pics sur l'ensemble des données
                        try:
                            min_distance = int(self.min_distance_period * self.AS7262_sample_rate)
                            peaks, _ = find_peaks(
                                self.data, 
                                distance=min_distance,
                                prominence=0.3,
                                height=0
                            )
                            
                            # Pour chaque pic, trouver le premier instant de la valeur maximale
                            first_max_indices = []
                            for peak_idx in peaks:
                                peak_value = self.data[peak_idx]
                                search_window = self.search_window_size
                                start_search = max(0, peak_idx - search_window)
                                
                                for i in range(start_search, peak_idx + 1):
                                    if self.data[i] == peak_value:
                                        first_max_indices.append(i)
                                        break
                            
                            # Créer une colonne pour les points Algo TC
                            algo_tc_column = np.zeros(len(self.data))
                            
                            # Pour chaque pic, chercher la dernière décélération dans la fenêtre
                            window_size_points = int(self.tc_window_size * self.AS7262_sample_rate)
                            
                            for peak_idx in first_max_indices:
                                end_search_idx = min(len(self.data) - 1, peak_idx + window_size_points)
                                
                                # Chercher la dernière décélération
                                last_deceleration_idx = None
                                for i in range(peak_idx, end_search_idx):
                                    if i < len(derivatives) and derivatives[i] < 0:
                                        last_deceleration_idx = i
                                
                                if last_deceleration_idx is not None:
                                    algo_tc_column[last_deceleration_idx] = 1
                            
                            data_dict["Algo_TC (1=détecté)"] = algo_tc_column
                        except Exception as e:
                            print(f"Erreur lors de la détection Algo TC pour l'export CSV: {e}")
                
                # Création du DataFrame
                # Première ligne: infos du fichier
                first_row = pd.DataFrame([header_info])
                
                # Données principales
                data_df = pd.DataFrame(data_dict)
                
                # Écrire dans le fichier CSV
                first_row.to_csv(csvfile, index=False)
                csvfile.write("\n")  # Ligne vide entre les infos et les données
                data_df.to_csv(csvfile, index=False)
                
                QMessageBox.information(self, "Export réussi", f"Les données ont été exportées avec succès dans {filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Une erreur est survenue lors de l'export: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = AS7262Logger()
    window.show()
    sys.exit(app.exec_())
