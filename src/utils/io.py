from tkinter import Tk, filedialog
from pickle import dump


def select_directory_dialog() -> str:
    selected_dir = filedialog.askdirectory()
    return selected_dir


def select_file_dialog() -> str:
    file = filedialog.askopenfile()
    return file.name


def save_file_dialog() -> str:
    save_file = filedialog.asksaveasfile()
    return save_file.name


def write_pickle(fn: str, obj: object):
    with open(fn, 'wb') as write_file:
        dump(obj, write_file)


def load_pickle(fn: str) -> object:
    with open(fn, 'rb') as load_file:
        obj = load_pickle(load_file)
    return obj
