import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

# QT5 imports
from PyQt5.QtWidgets import (QMainWindow, QMenu, QMessageBox, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget, QSizePolicy, QFileDialog, QPushButton, QListWidget)

from PyQt5.QtCore import pyqtSlot, Qt


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        grid = gridspec.GridSpec(6, 2)
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = fig.add_subplot(grid[:-1, 0])
        self.ax2 = fig.add_subplot(grid[0:3, 1])
        self.ax3 = fig.add_subplot(grid[3:, 1], label='x')
        self.ax4 = fig.add_subplot(grid[3:, 1], label='y', frameon=False)  # This creates a second plot on top of ax3. This way, we can plot multiple stuff at the same location, while still controlling the behaviour individually
        cbax1 = fig.add_subplot(grid[-1, 0])
        fig.subplots_adjust(wspace=0.3, hspace=2.2)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""

    def compute_initial_figure(self):
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2*np.pi*t)
        self.ax1.plot(t, s)


class DirView(QWidget):

    def __init__(self, lof):
        super(QWidget, self).__init__()
        s = QPushButton('Science')
        s.clicked.connect(self.on_click)
        b = QPushButton('Bias')
        t = QPushButton('ThAr')
        f = QPushButton('Flat')

        glayout = QHBoxLayout()
        type = QVBoxLayout()
        type.addWidget(s)
        type.addWidget(b)
        type.addWidget(t)
        type.addWidget(f)
        details = QVBoxLayout()
        self.content = QListWidget()

        details.addWidget(self.content)

        glayout.addLayout(type)
        glayout.addLayout(details)
        self.setLayout(glayout)

    @pyqtSlot()
    def on_click(self):
        print('bouton poussé')
        self.content.addItem('test')


class HRS_Window(QWidget):
    def __init__(self, parent, directory):
        super(QWidget, self).__init__(parent)
        self.directory = directory
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab2 = MyStaticMplCanvas()
        self.tab1 = DirView(directory)
        self.tabs.addTab(self.tab1, self.directory)
        self.tabs.addTab(self.tab2, "HRS Frame")
        self.layout.addWidget(self.tabs)


class MainWindow(QMainWindow):

    def __init__(self, model=None, directory='toto'):
        QMainWindow.__init__(self)
        self.directory= directory
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("HRS Data Reduction Pipeline")

        self.file_menu = QMenu('&File', self)
        self.file_menu.addAction('&Choose directory', self.opendirectoryDialog, Qt.CTRL+Qt.Key_O)
        self.file_menu.addAction('&Quit', self.fileQuit, Qt.CTRL + Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        # self.main_widget = QtWidgets.QWidget(self)
        self.main_widget = HRS_Window(self, self.directory)

        # l = QtWidgets.QVBoxLayout(self.main_widget)
        # sc = MyStaticMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        # dc = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        # l.addWidget(sc)
        # l.addWidget(dc)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage(self.directory, 2000)

    def opendirectoryDialog(self):
        self.selecteddir = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.statusBar().showMessage(self.selecteddir)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QMessageBox.about(self, "About",
                                """Embedding_in_qt5.py example
Copyright 2005 Florent Rougon, 2006 Darren Dale, 2015 Jens H Nielsen.\nThis program is a simple example of a Qt5 application embedding matplotlib canvases.\nIt may be used and modified with no restriction; raw copies as well as
modified versions may be distributed without limitation.\nThis is modified from the embedding in qt4 example to show the difference between qt4 and qt5""")
