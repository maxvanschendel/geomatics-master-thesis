from tkinter import Tk, filedialog
from pickle import dump, load


def select_directory() -> str:
    selected_dir = filedialog.askdirectory()
    return selected_dir


def select_file() -> str:
    file = filedialog.askopenfile()
    return str(file.name)


def save_file_dialog() -> str:
    save_file = filedialog.asksaveasfile()
    return save_file.name


def write_pickle(fn: str, obj: object):
    with open(fn, 'wb') as write_file:
        dump(obj, write_file)


def load_pickle(fn: str) -> object:
    with open(fn, 'rb') as load_file:
        obj = load(load_file)
    return obj
