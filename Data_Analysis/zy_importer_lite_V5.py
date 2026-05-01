# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:42:43 2023
1. python version >3.10 (don't use python installed from anaconda, need tcl/tk')
2. useful data would be in csv column 1,3 +10*i
3. add filter for file_list: include and exclude; done 8/16/23
@author: Zengyou Ye

"""

import tkinter as tk
from tkinter import filedialog
import os

class FileSelectorWindow:
    def __init__(self):
        # --- Default Settings ---
        self.extension_choices = ["csv", "xlsx", "txt", "doric", "tdms", "tif"]
        self.default_path = 'C:/Users/addy/Documents/GitHub/Matlab_photometry/Matlab_photometry'
        self.extension_choice = self.extension_choices[0]
        self.include_subfolders = False
        
        # --- Output Variables ---
        self.files = []
        self.selected_files = []
        self.bonsai_file = False
        self.user_cancelled = False

    def select(self):
        # 1. Initialize Main Window FIRST
        self.root = tk.Tk()
        self.root.title("File_selector GUI Lite V5 by Zengyou")
        self.root.attributes("-topmost", True)

        # 2. Setup Tkinter Variables (Crucial: Must pass self.root to prevent blank fields!)
        self.folder_var = tk.StringVar(self.root, value=self.default_path)
        self.selected_extension = tk.StringVar(self.root, value=self.extension_choice)
        self.include_subfolders_var = tk.BooleanVar(self.root, value=self.include_subfolders)
        self.bonsai_file_var = tk.BooleanVar(self.root, value=self.bonsai_file)
        self.include_var = tk.StringVar(self.root, value="")
        self.exclude_var = tk.StringVar(self.root, value="summary")

        # 3. Build UI & Populate
        self.build_ui()
        self.update_file_list()

        # 4. Handle Window Close ('X' button)
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)
        
        # 5. Start Loop
        self.root.mainloop()

    def build_ui(self):
        # --- Folder Frame ---
        folder_frame = tk.Frame(self.root)
        folder_frame.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        
        tk.Label(folder_frame, text="1. Select a folder").grid(row=0, column=0, sticky='w')
        tk.Label(folder_frame, text="file ext:").grid(row=1, column=1, sticky='e')
        
        tk.OptionMenu(folder_frame, self.selected_extension, *self.extension_choices, 
                      command=self.update_file_list).grid(row=1, column=2, sticky='w')
        
        tk.Entry(folder_frame, textvariable=self.folder_var, width=70).grid(row=2, column=0, columnspan=2, sticky='w')
        tk.Button(folder_frame, text="Browse", command=self.browse_folder).grid(row=3, column=0, sticky='w', pady=2)

        # --- Filter Frame ---
        filter_frame = tk.Frame(self.root)
        filter_frame.grid(row=1, column=0, padx=10, pady=5, sticky='w')
        
        tk.Label(filter_frame, text="2. Filter").grid(row=0, column=0, sticky='w')
        
        tk.Checkbutton(filter_frame, text="Include Subfolders", variable=self.include_subfolders_var, 
                       command=self.update_file_list).grid(row=1, column=0, sticky='w')
        tk.Checkbutton(filter_frame, text="bonsai_file", variable=self.bonsai_file_var, 
                       fg='red').grid(row=1, column=2, sticky='w')
        
        tk.Label(filter_frame, text="Include in filename (,):").grid(row=2, column=0, sticky='w')
        tk.Entry(filter_frame, textvariable=self.include_var, width=50).grid(row=2, column=1, sticky='w')
        
        tk.Label(filter_frame, text="Exclude in filename (,):").grid(row=3, column=0, sticky='w')
        tk.Entry(filter_frame, textvariable=self.exclude_var, width=50).grid(row=3, column=1, sticky='w')

        self.include_var.trace_add("write", self.update_file_list)
        self.exclude_var.trace_add("write", self.update_file_list)

        # --- List Frame ---
        file_frame = tk.Frame(self.root)
        file_frame.grid(row=2, column=0, padx=10, pady=5, sticky='w')
        
        tk.Label(file_frame, text="3. Select Files:").grid(row=0, column=0, sticky='w')
        
        scrollbar = tk.Scrollbar(file_frame, orient=tk.VERTICAL)
        self.file_listbox = tk.Listbox(file_frame, selectmode=tk.MULTIPLE, width=70, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.file_listbox.yview)
        
        self.file_listbox.grid(row=1, column=0, sticky='w')
        scrollbar.grid(row=1, column=1, sticky='ns')

        tk.Button(file_frame, text="Select All", command=self.select_all).grid(row=2, column=0, sticky='w', pady=2)

        # --- Buttons Frame ---
        bottom_frame = tk.Frame(self.root)
        bottom_frame.grid(row=3, column=0, padx=10, pady=10, sticky='e')
        
        self.warning_label = tk.Label(bottom_frame, text='', fg='red')
        self.warning_label.grid(row=0, column=0, padx=10)
        
        tk.Button(bottom_frame, text="Next", command=self.on_run).grid(row=0, column=1, padx=5)
        tk.Button(bottom_frame, text="Quit", command=self.on_quit).grid(row=0, column=2, padx=5)

    def browse_folder(self):
        folder_path = filedialog.askdirectory(parent=self.root, title='Select Folder', initialdir=self.folder_var.get())
        if folder_path:
            self.folder_var.set(folder_path)
            self.update_file_list()

    def update_file_list(self, *args):
        self.file_listbox.delete(0, tk.END)
        self.files.clear()
        
        folder_path = self.folder_var.get()
        if not os.path.isdir(folder_path):
            return

        inc_strs = [s.strip().lower() for s in self.include_var.get().split(',') if s.strip()]
        exc_strs = [s.strip().lower() for s in self.exclude_var.get().split(',') if s.strip()]
        ext = self.selected_extension.get().lower()

        def should_include(filename):
            name_lower = filename.lower()
            if not name_lower.endswith(ext): return False
            if inc_strs and not all(s in name_lower for s in inc_strs): return False
            if exc_strs and any(s in name_lower for s in exc_strs): return False
            return True

        if self.include_subfolders_var.get():
            for root_path, _, filenames in os.walk(folder_path):
                for file in filenames:
                    if should_include(file):
                        self.file_listbox.insert(tk.END, file)
                        self.files.append(os.path.join(root_path, file))
        else:
            for file in os.listdir(folder_path):
                if should_include(file):
                    self.file_listbox.insert(tk.END, file)
                    self.files.append(os.path.join(folder_path, file))

    def select_all(self):
        self.file_listbox.select_set(0, tk.END)

    def on_run(self):
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            self.warning_label.config(text="Warning: No file selected")
            return

        self.selected_files = [self.files[i] for i in selected_indices]
        self.bonsai_file = self.bonsai_file_var.get()
        self.user_cancelled = False
        
        # Safe Exit Sequence
        self.root.quit()
        self.root.destroy()

    def on_quit(self):
        print('User quit the selector')
        self.user_cancelled = True
        self.selected_files = []
        
        # Safe Exit Sequence
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    file_selector = FileSelectorWindow()
    file_selector.select()
    
    if file_selector.user_cancelled:
        print("Script stopped cleanly.")
    else:
        print(f">>> Are they bonsai file: {file_selector.bonsai_file}")
        print(">>> Number of files: ", len(file_selector.selected_files))
        print(*file_selector.selected_files, sep='\n   ')        
        print("----------")


