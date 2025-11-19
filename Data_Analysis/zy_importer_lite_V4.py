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
import os,sys

#%% like class!
class FileSelectorWindow:
    def __init__(self):
        self.extension_choices = ["csv", "xlsx", "txt", "doric","tdms", "tif"]
        self.default_path = 'C:/Users/addy/Documents/GitHub/Matlab_photometry/Matlab_photometry'
        self.extension_choice = self.extension_choices[0]
        self.include_subfolders = False
        # preset values for output
        self.files=[]
        self.selected_files = []
        self.bonsai_file = False
           
    #%% select files     
    def select(self):    

        def get_selected_files(file_list):
            selected_files = file_list.curselection()
            return [self.files[index] for index in selected_files]
        #%%      
        def update_file_list(folder_path):
            file_list.delete(0, tk.END)  # Clear previous items in the Listbox
            self.files.clear()
            
            include_strings = [s.strip() for s in include_var.get().split(',') if s.strip()]
            exclude_strings = [s.strip() for s in exclude_var.get().split(',') if s.strip()]
            file_ext = selected_extension.get()
            # aligned_file_ext = '_aligned.' + file_ext
            
            def should_include_file(file_name):
                check = True
        
                if include_strings and not all(include_str in file_name for include_str in include_strings):
                    check = False
                
                if exclude_strings and any(exclude_str in file_name for exclude_str in exclude_strings):
                    check = False
                
                return check
    
            if os.path.isdir(folder_path):
                if include_subfolders_var.get():
                    for root_path, _, filenames in os.walk(folder_path):
                        for file in filenames:
                            if (file.endswith(file_ext) and should_include_file(file)):
                                file_list.insert(tk.END, file)
                                self.files.append(os.path.join(root_path, file))
                else:
                    for file in os.listdir(folder_path):
                        if (file.endswith(file_ext) and should_include_file(file)):
                            file_list.insert(tk.END, file)
                            self.files.append(os.path.join(folder_path, file))
    
        def select_all():
            file_list.select_set(0, tk.END)
    
        def on_checkbox_toggle():
            folder_path = folder_var.get()
            update_file_list(folder_path)
        
        def on_dropdown_change(value):
            folder_path = folder_var.get()
            update_file_list(folder_path)
    

        # files=[]
        root = tk.Tk()  # Create the main window
        # root.geometry('420x420')
        root.attributes("-topmost", True)
        root.title("File_selector GUI Lite V4 by Zengyou")
        
        #%% Folder Selection Frame
        folder_frame = tk.Frame(root)
        folder_frame.grid(row=0, column=0, padx=5, pady=10, sticky='w')

        folder_label = tk.Label(folder_frame, text="1. Select a folder")
        folder_label.grid(row=0, column=0, sticky='w')  # west
        
        # Dropdown Menu for Selecting Extension
        extension_label = tk.Label(folder_frame, text="file ext:")
        extension_label.grid(row=1, column=0, sticky='e')

        selected_extension = tk.StringVar()
        selected_extension.set(self.extension_choice)  # Default 1st extension
        extension_dropdown = tk.OptionMenu(
            folder_frame, selected_extension, *self.extension_choices,command=on_dropdown_change)
        extension_dropdown.grid(row=1, column=1, sticky='w')

        folder_var = tk.StringVar()
        folder_var.set(self.default_path)  # Set the default_folder
        folder_entry = tk.Entry(
            folder_frame, textvariable=folder_var, width=70)
        folder_entry.grid(row=2, column=0, sticky='w')
        
        def browse_folder():
            folder_path = filedialog.askdirectory(title='Select Folder with csv files',
                                                       initialdir=self.default_path)
            if folder_path:
                folder_var.set(folder_path)
                update_file_list(folder_path)
                
        folder_button = tk.Button(
            folder_frame, text="Browse", command=browse_folder)
        folder_button.grid(row=3, column=0, sticky='w')

        #%% File filter Frame
        filter_frame = tk.Frame(root)
        filter_frame.grid(row=1, column=0, padx=5, pady=10, sticky='w')

        folder_label = tk.Label(filter_frame, text="2. Filter")
        folder_label.grid(row=0, column=0, sticky='w')  # west
        
        # Checkbox for Including Subfolders
        include_subfolders_var = tk.BooleanVar()
        include_subfolders_var.set(self.include_subfolders)  # Default is not including subfolders
        include_subfolders_checkbox = tk.Checkbutton(
            filter_frame, text="Include Subfolders", 
            variable=include_subfolders_var, command=on_checkbox_toggle)
        include_subfolders_checkbox.grid(row=1, column=0, sticky='w')
        
        # Checkbox for other problem to determine whether to run in Bonsai way
        bonsai_file_var = tk.BooleanVar()
        bonsai_file_var.set(self.bonsai_file)  # Default is not including subfolders
        bonsai_file_checkbox = tk.Checkbutton(
            filter_frame, text="bonsai_file", 
            variable=bonsai_file_var, command=on_checkbox_toggle,fg='red')
        bonsai_file_checkbox.grid(row=1, column=2, sticky='w')
        
        include_label = tk.Label(filter_frame, text="Include in filename (,):")
        include_label.grid(row=2, column=0, sticky='w')
        
        include_var = tk.StringVar()
        # include_var.set('ZY')
        
        include_entry = tk.Entry(filter_frame, textvariable=include_var, width=50)
        include_entry.grid(row=2, column=1, sticky='w')
        
        exclude_label = tk.Label(filter_frame, text="Exclude in filename (,):")
        exclude_label.grid(row=3, column=0, sticky='w')
        
        exclude_var = tk.StringVar()
        exclude_var.set('summary')
        exclude_entry = tk.Entry(filter_frame, textvariable=exclude_var, width=50)
        exclude_entry.grid(row=3, column=1, sticky='w')
        
        # Update the file list whenever inclusion or exclusion strings change
        include_var.trace_add("write", lambda *args: update_file_list(folder_var.get()))
        exclude_var.trace_add("write", lambda *args: update_file_list(folder_var.get()))
        
        #%% File List Frame
        file_frame = tk.Frame(root)
        file_frame.grid(row=2, column=0, padx=5, pady=10, sticky='w')

        file_label = tk.Label(file_frame, text="3. Select Files:")
        file_label.grid(row=0, column=0, sticky='w')

        file_list = tk.Listbox(
            file_frame, selectmode=tk.MULTIPLE, width=70)
        file_list.grid(row=1, column=0, sticky='w')
        update_file_list(self.default_path)

        select_all_button = tk.Button(
            file_frame, text="Select All", command=select_all)
        select_all_button.grid(row=2, column=0, sticky='w')

        #%% Run and Cancel Buttons Frame
        def update_value():
            selected_files = get_selected_files(file_list)
            # save those 4 important parameters
            self.selected_files = selected_files
            self.bonsai_file = bonsai_file_var.get()
            
        def preview():
            print(f">>> Are they bonsai file: {self.bonsai_file}")
            print(">>> Number of selected_files: ", len(self.selected_files))
            print(*[file for file in self.selected_files], sep='\n   ')        
            print("----------")
                        
        def show_warning():
            warning_label.config(text="Warning: No file selected",fg='red')
            
        def on_run_button_click():
            update_value()
            if not self.selected_files:
                show_warning()
            else:
                preview()

                # add your functional output if needed
                print("Destroying root after clicking Run button")
                root.destroy()  # close the GUI, use root.quit() or destroy()? tried!
            
        def on_quit_button_click():  # need to clear back to default
            print('User quit the selector')
            root.destroy()
            sys.exit()  # stop the Python script
        
        button_frame = tk.Frame(root)
        button_frame.grid(row=3, column=0, padx=5, pady=5, sticky='e')        
        
        run_button = tk.Button(
            button_frame, text="Next", command=on_run_button_click)
        run_button.grid(row=0, column=0, sticky='w',padx=10)

        quit_button = tk.Button(
            button_frame, text="Quit", command=on_quit_button_click)
        quit_button.grid(row=0, column=1, sticky='w',padx=10)
        
        #%% warning frame
        warning_frame = tk.Frame(root)
        warning_frame.grid(row=4, column=0, padx=5, pady=5, sticky='e')   
        
        warning_label = tk.Label(warning_frame, text='')
        warning_label.grid(row=0, column=0, sticky='w', padx=10)

        root.protocol("WM_DELETE_WINDOW", on_quit_button_click)        
        root.mainloop()  # Keep the root GUI active
         
#%% Example usage of the FileSelectorWindow class
if __name__ == "__main__":
    file_selector = FileSelectorWindow()
    # option

    file_selector.select()
    
    file_list = file_selector.selected_files

    print(f">>> Are they bonsai file: {file_selector.bonsai_file}")
    print(">>> Number of files: ", len(file_list))
    print(*[file for file in file_list], sep='\n   ')        

    print("----------") 
