from pathlib import Path
from pipeline import ListOfFiles


class Controller(object):
    def __init__(self, directory):
        self.directory = directory
        lof = ListOfFiles(self.directory)
