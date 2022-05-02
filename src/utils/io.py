from tkinter import Tk, filedialog


def select_directory_dialog():
    root = Tk()
    root.withdraw()
    selected_dir = filedialog.askdirectory()
    return selected_dir

def select_file_dialog():
    file = filedialog.askopenfile()
    return file

def save_file_dialog():
    save_file = filedialog.asksaveasfile()
    return save_file