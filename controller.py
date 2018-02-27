from pathlib import Path
from pipeline import ListOfFiles, MainWindow
from pipeline import Data


class Controller(object):
    def __init__(self, directory='./'):
        self.model = Data()
        self.main_view = MainWindow(model=self.model, directory='/home/eric/Temp') #, self.main_ctrl)
        self.main_view.show()
    def on_click(self):
        print('click')
