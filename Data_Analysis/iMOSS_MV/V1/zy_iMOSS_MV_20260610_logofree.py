# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 16:25:54 2025
*****************
This Video Frame Scoring script is a user-friendly, open-source software designed to
manually score Binary Behaviors like mobility and immobility in experiments such as the 
Tail Suspension Test (TST) and Forced Swim Test (FST). Researchers can 
efficiently annotate video frames, categorizing behavior as either mobility 
or immobility while maintaining precise frame control.
********************

20250624: update fill_frame_time to read systemTime from FP3002 if exist
@author: Zengyou Ye at NIH/NIDA/IRP (addy9908@gmail.com)
"""

import time, os, io
import re
import shutil
import cv2
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, Canvas, messagebox, font
from collections import deque
from PIL import Image, ImageTk
from pyexcelerate import Workbook
import warnings
warnings.simplefilter("ignore", UserWarning)

import base64
# Import the function from the file you just generated!
LOGO_BASE64 = "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"

def get_logo():
    image_data = base64.b64decode(LOGO_BASE64)
    image_stream = io.BytesIO(image_data)
    img = Image.open(image_stream)
    img.load()
    return img

class VideoScoring:
    def __init__(self):
        self.version = 'zy_iMOSS_MV_20251119.py'
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.frame_number = 0
        self.playing = False
        self.original_fps = 0
        self.play_fps = 30
        self.session_duration = 360
        self.data = {}
        self.rois = {}
        self.loaded_rois = {}
        self.loaded_data = {}
        self.current_mouse_id = None
        self.frame_cache = deque(maxlen=28000)
        self.last_immobility = "Skip"
        self.help_visible = False
        self.bin_size = 10
        self.total_immobility = 0 #sec, don't do this, bad for resume scoring
        self.auto_save = False
        self.roi_frame = None
        
        self.debug = 0
        
        self.init_ui()
    
    def init_ui(self):
        self.tk_window = tk.Tk()
        self.tk_window.title("iMOSS-MV: Immobility & Mobility Optimized Scoring System – Video-based")
        self.tk_window.resizable(True, True)
        self.tk_window.bind("<space>", self.toggle_play)
        self.tk_window.bind("<Right>", self.next_frame)
        self.tk_window.bind("<Left>", self.prev_frame)
        self.tk_window.bind("r", self.select_roi)
        self.tk_window.bind("<Key-0>", lambda event: self.on_number_key(event, 0))
        self.tk_window.bind("<Key-1>", lambda event: self.on_number_key(event, 1))
        self.tk_window.bind("h", self.toggle_help)
        self.tk_window.bind("<Delete>", self.remove_immobility_data)
        bold_font = font.Font(weight="bold")
        file_frame = tk.Frame(self.tk_window)
        file_frame.pack()
        try:
            # img = Image.open('iMOSS-V_new.png')
            # img = img.resize((200,40),Image.LANCZOS)
            img = get_logo()
            logo = ImageTk.PhotoImage(img)
            logo_main = tk.Label(
                file_frame,
                image = logo
            )
            logo_main.image = logo
        except:
            logo_main = tk.Label(
                file_frame,
                text="📹 iMOSS-MV 🐁",
                font=("Helvetica", 14, "bold","italic"),
                fg="#004080",
                bg="#f2f2f2"
            )
            
        logo_main.grid(row=0, column=0, sticky=tk.E, padx=10)
        # tk.Label(file_frame, text="Video File:",font = bold_font).pack(side=tk.LEFT, padx=5)
        tk.Button(file_frame, text="Load video", command=self.load_video,font = bold_font,bg="red", fg="white",).grid(row=0, column=1, sticky=tk.E, padx=5)
        self.file_entry = tk.Entry(file_frame, width=60, justify='right')
        self.file_entry.grid(row=0, column=2, sticky=tk.E, padx=5)
        
        
        self.timer_label = tk.Label(file_frame, text="Time (s): N/A", font = bold_font, fg="red",width=40)
        self.timer_label.grid(row=0, column=3, sticky=tk.E, padx=5)
        
        first_frame = tk.Frame(self.tk_window)
        first_frame.pack(pady=5)
        
        tk.Button(first_frame, text="Draw ROI (r)", command=self.select_roi, bg="orange", fg="white", font = bold_font,width=20).grid(row=0, column=0, sticky=tk.E, padx=5)
        tk.Button(first_frame, text="Resume Scoring", command=self.resume_scoring, bg="orange", fg="white", font = bold_font,width=20).grid(row=0, column=1, sticky=tk.E, padx=5)
        tk.Button(first_frame, text="Remove Frame Data", command=self.remove_immobility_data, bg="orange", fg="white", font = bold_font,width=20).grid(row=0, column=2, sticky=tk.E, padx=5)

        self.speed_entry = tk.Entry(first_frame, width=20)
        self.speed_entry.grid(row=0, column=3, sticky=tk.E, padx=5)
        self.speed_entry.insert(0, self.play_fps)
        tk.Button(first_frame, text="Set fps", command=self.set_play_fps, font = bold_font,width=20).grid(row=0, column=4, sticky=tk.E, padx=5)
        
        tk.Button(first_frame, text="Shortcut List (H)", command=self.toggle_help,font = bold_font,width=20).grid(row=0, column=5, sticky=tk.E, padx=5)
        
        second_frame = tk.Frame(self.tk_window)
        second_frame.pack(pady=5)
        
        tk.Button(second_frame, text="⏮", command=self.prev_frame, bg="blue", fg="white", font = bold_font,width=20).grid(row=0, column=0, sticky=tk.E, padx=5)
        self.play_pause_btn = tk.Button(second_frame, text="▶", command=self.toggle_play, bg="blue", fg="white", font = bold_font,width=20)
        self.play_pause_btn.grid(row=0, column=1, sticky=tk.E, padx=5)
        tk.Button(second_frame, text="⏭", command=self.next_frame, bg="blue", fg="white", font = bold_font,width=20).grid(row=0, column=2, sticky=tk.E, padx=5)

        self.frame_entry = tk.Entry(second_frame, width=20)
        self.frame_entry.grid(row=0, column=3, sticky=tk.E, padx=5)
        self.frame_entry.insert(0, 0)

        jump_button = tk.Button(second_frame, text="Jump to Frame", command=lambda: self.jump_to_frame(int(self.frame_entry.get())), font = bold_font,width=20)
        jump_button.grid(row=0, column=4, sticky=tk.E, padx=5)                     
        
        self.mobility_button = tk.Button(second_frame, text="Mobility (0)", command=lambda: self.mark_immobility(0), bg = 'black', fg = 'white', font = bold_font, width=20)
        self.mobility_button.grid(row=0, column=5, sticky=tk.E, padx=5)

        
        third_frame = tk.Frame(self.tk_window)
        third_frame.pack(pady=5)

        
        tk.Button(third_frame, text="Save ROI Image", command=self.save_current_roi_image, bg="purple", fg="white",font = bold_font,width=20).grid(row=0, column=0, sticky=tk.E, padx=5)
        tk.Button(third_frame, text="Save ROI mp4", command=self.save_video_segment, bg="purple", fg="white",font = bold_font,width=20).grid(row=0, column=1, sticky=tk.E, padx=5)
        tk.Button(third_frame, text="Save & Next Mouse", command=self.save_and_next_mouse, bg="purple", fg="white", font = bold_font,width=20).grid(row=0, column=2, sticky=tk.E, padx=5)
        tk.Button(third_frame, text="Save Data", command=self.save_data, bg="purple", fg="white", font = bold_font,width=20).grid(row=0, column=3, sticky=tk.E, padx=5)
        
        spacer = tk.Label(third_frame, text="", width=20)  # or use tk.Frame with width
        spacer.grid(row=0, column=4,padx=5)
        
        self.immobility_button = tk.Button(third_frame, text="Immobility (1)", command=lambda: self.mark_immobility(1), bg = 'black', fg = 'white',font = bold_font, width=20)
        self.immobility_button.grid(row=0, column=5, sticky=tk.E, padx=5)
       
        self.canvas = Canvas(self.tk_window)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.tk_window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        def unfocus_if_not_entry(event):
            widget = event.widget
            if not isinstance(widget, tk.Entry):
                self.tk_window.focus_set()
                
        self.tk_window.bind("<Button-1>", unfocus_if_not_entry) # move cursor out of entries
    

    def save_video_segment(self):
        # Get start and end frame numbers
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        
        if self.rois.get(self.current_mouse_id) and self.data[self.current_mouse_id]:
            x, y, w, h = self.rois[self.current_mouse_id]          

            min_frame = min(self.data[self.current_mouse_id].keys())
            end_frame = int(min_frame + int(self.session_duration*self.original_fps)) #6 min at 
        
            # Get original video basename
            mp4_path = f"{self.video_path.rsplit('.', 1)[0]}_{self.current_mouse_id}_{timestamp}.mp4"
            # Define VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') #MJPG, *'mp4v',MPEG *'XVID'*'MJPG'
            out = cv2.VideoWriter(mp4_path, fourcc, self.original_fps, (w, h))
        
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame) # get original color video
        
            for frame_number in range(min_frame, end_frame):           
                # self.frame_number = frame_number
                ret, frame = self.cap.read()
                if not ret:
                    return
                    # print(f"Warning: Frame {frame_number} not in cache, skipping.")
                      
                roi_frame = frame[y:y + h, x:x + w]
                
                # get time infor
                first_mobility_frame = min(self.data[self.current_mouse_id].keys())
                time_passed = round((frame_number - first_mobility_frame) / self.original_fps,2)
    
                frame_number_text = f"Frame: {frame_number:05d} ({time_passed:.2f}s)"
                cv2.putText(roi_frame, frame_number_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                if self.current_mouse_id in self.data: # and self.data[self.current_mouse_id]: #the get_immobility_label can take care of empty data[mouse_id]
                    immobility = self.get_immobility_label(self.data[self.current_mouse_id], frame_number)
                    color = (0,0,255) if 'Immobility' in immobility else (0,255,0)
                    cv2.putText(roi_frame, f"{self.current_mouse_id}: {immobility}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                out.write(roi_frame)
                     
            out.release()
            print(f"Video saved to: {mp4_path}")
            #release and reopen
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            self.frame_number = 0
            self.process_frame()

    def save_video_segment_old(self):
        # Get start and end frame numbers
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        x, y, w, h = self.rois[self.current_mouse_id]
        
        min_frame = min(self.data[self.current_mouse_id].keys())
        end_frame = int(min_frame + int(self.session_duration*self.original_fps)) #6 min at 
    
        # Get original video basename
        mp4_path = f"{self.video_path.rsplit('.', 1)[0]}_{self.current_mouse_id}_{timestamp}.mp4"
    
        # Sort frame cache into a dictionary for faster access
        cache_dict = dict(self.frame_cache)
    
    
        # Define VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') #MJPG, *'mp4v',MPEG *'XVID'*'MJPG'
        out = cv2.VideoWriter(mp4_path, fourcc, self.original_fps, (w, h))
    
        for frame_number in range(min_frame, end_frame):
            frame = cache_dict.get(frame_number)
            if frame is None: 
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = self.cap.read()
                if not ret:
                    continue
                # print(f"Warning: Frame {frame_number} not in cache, skipping.")
            else:    
                frame = self.decompress_frame_jpg(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
            roi_frame = frame[y:y + h, x:x + w]

            frame_number_text = f"Frame: {frame_number}"
            cv2.putText(roi_frame, frame_number_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            if self.current_mouse_id in self.data: # and self.data[self.current_mouse_id]: #the get_immobility_label can take care of empty data[mouse_id]
                immobility = self.get_immobility_label(self.data[self.current_mouse_id], frame_number)
                color = (0,0,255) if 'Immobility' in immobility else (0,255,0)
                cv2.putText(roi_frame, f"{self.current_mouse_id}: {immobility}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(roi_frame)
                 
        out.release()
        print(f"Video saved to: {mp4_path}")

    def save_video_segment_avi(self):
        # Get start and end frame numbers
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        x, y, w, h = self.rois[self.current_mouse_id]
        
        min_frame = min(self.data[self.current_mouse_id].keys())
        end_frame = int(min_frame + int(self.session_duration*self.original_fps)) #6 min at 
    
        # Get original video basename
        mp4_path = f"{self.video_path.rsplit('.', 1)[0]}_{self.current_mouse_id}_{timestamp}.avi"
    
        # Sort frame cache into a dictionary for faster access
        cache_dict = dict(self.frame_cache)
    
    
        # Define VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID') #MJPG, *'mp4v',MPEG *'XVID'*'MJPG'
        out = cv2.VideoWriter(mp4_path, fourcc, self.original_fps, (w, h))
    
        for frame_number in range(min_frame, end_frame): # may save from the 1st frame in the future
            frame = cache_dict.get(frame_number)
            if frame is None:
                # print(f"Warning: Frame {frame_number} not in cache, skipping.")
                continue
    

            frame = self.decompress_frame_jpg(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            roi_frame = frame[y:y + h, x:x + w]

            
            frame_number_text = f"Frame: {frame_number}"
            cv2.putText(roi_frame, frame_number_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            if self.current_mouse_id in self.data: # and self.data[self.current_mouse_id]: #the get_immobility_label can take care of empty data[mouse_id]
                immobility = self.get_immobility_label(self.data[self.current_mouse_id], frame_number)
                color = (0,0,255) if 'Immobility' in immobility else (0,255,0)
                cv2.putText(roi_frame, f"{self.current_mouse_id}: {immobility}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
            # Convert to BGR (3 channels)
            
            out.write(roi_frame)
                 
        out.release()
        print(f"Video saved to: {mp4_path}")

    
    def save_current_roi_image(self):   
        # Compose filename
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        save_path = f"{self.video_path.rsplit('.', 1)[0]}_{self.current_mouse_id}_{self.frame_number}.tif"
        

        if self.rois.get(self.current_mouse_id) and self.data[self.current_mouse_id]:
            x, y, w, h = self.rois[self.current_mouse_id]          

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            ret, frame = self.cap.read()
            if not ret:
                return
                # print(f"Warning: Frame {frame_number} not in cache, skipping.")
                  
            roi_frame = frame[y:y + h, x:x + w]
            roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            
            # get time infor
            first_mobility_frame = min(self.data[self.current_mouse_id].keys())
            time_passed = round((self.frame_number - first_mobility_frame) / self.original_fps,2)

            frame_number_text = f"Frame: {self.frame_number:05d} ({time_passed:.2f}s)"
            cv2.putText(roi_frame, frame_number_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, w/400, (0, 0, 255), 2)
            
            if self.current_mouse_id in self.data: # and self.data[self.current_mouse_id]: #the get_immobility_label can take care of empty data[mouse_id]
                immobility = self.get_immobility_label(self.data[self.current_mouse_id], self.frame_number)
                color = (255,0,0) if 'Immobility' in immobility else (0,255,0)
                cv2.putText(roi_frame, f"{self.current_mouse_id}: {immobility.split()[0]}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, w/400, color, 2) # size: 300w as 0.75
                
            image_pil = Image.fromarray(roi_frame) 
            # Save with DPI = 300
            image_pil.save(save_path, dpi=(600, 600))
        
            # # Save image
            # cv2.imwrite(save_path, cv2.cvtColor(self.roi_frame, cv2.COLOR_RGB2BGR)) #cv treat image as BGR
            print(f"Saved: {save_path}")
        
    def save_current_roi_image_old(self):   
        # Compose filename
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        save_path = f"{self.video_path.rsplit('.', 1)[0]}_{self.current_mouse_id}_{self.frame_number}.tif"
        
        image_pil = Image.fromarray(self.roi_frame) 

        # Save with DPI = 300
        image_pil.save(save_path, dpi=(300, 300))
    
        # # Save image
        # cv2.imwrite(save_path, cv2.cvtColor(self.roi_frame, cv2.COLOR_RGB2BGR)) #cv treat image as BGR
        print(f"Saved: {save_path}")        
    
    def on_close(self):
        self.frame_cache.clear()
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.tk_window.destroy()
        print('User close the GUI')
    
    def set_play_fps(self):
        # self.tk_window.focus()
        try:
            self.play_fps = max(1, int(self.speed_entry.get()))
        except ValueError:
            print('Use original rate')
            self.play_fps = self.original_fps

    def jump_to_frame(self, target_frame):
        # self.tk_window.focus()
        # Get the frame number from the entry field
        # target_frame = int(self.frame_entry.get())
        
        # Ensure the target frame is within valid bounds
        if 0 <= target_frame < self.total_frames:
            self.frame_number = target_frame
            if self.frame_number in [f[0] for f in self.frame_cache]:
                frame = next(f[1] for f in self.frame_cache if f[0] == self.frame_number)
                frame = self.decompress_frame_jpg(frame)
                if self.debug:
                    print(f'Current frame: {self.frame_number}')  
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
                ret, frame = self.cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                jpg_bytes = self.compress_frame_jpg(frame, quality=80)
                self.frame_cache.append((self.frame_number, jpg_bytes))
                # self.frame_cache.append((self.frame_number, frame.copy()))
                if self.debug:
                    print(f'Current frame: {self.cap.get(cv2.CAP_PROP_POS_FRAMES)-1}')            
            
            if self.rois:
                self.update_canvas_with_roi(frame, self.rois[self.current_mouse_id])
            else:
                height, width = frame.shape[:2]
                self.update_canvas_with_roi(frame, (0,0,width,height))
        else:
            print(f"Invalid frame number. Please enter a value between 0 and {self.total_frames-1}.")
    
    def toggle_help(self, event=None):
        if not self.help_visible:
            self.help_window = tk.Toplevel(self.tk_window)
            self.help_window.title("Shortcut list")
            # self.help_window.geometry("300x400")
            self.help_window.resizable(False, False)

            
            help_text = (
                "Left Arrow   - Previous Frame ⏮\n"
                "Space        - Play/Pause\n"
                "Right Arrow  - Next Frame ⏭\n"
                "R            - Select ROI\n"
                "0            - Mark Mobility\n"
                "1            - Mark Immobility\n"
                "H            - Toggle Help\n"
                "Delete       - Remove frame data\n\n"
                "********************************\n"
                "Contact Author:\n"
                "  - Zengyou Ye\n"
                "  - addy9908@gmail.com"
            )
            
            help_label = tk.Label(self.help_window, text=help_text, justify=tk.LEFT, anchor="w")
            help_label.pack(padx=10, pady=10)
            self.help_visible = True
        else:
            self.help_window.destroy()
            self.help_visible = False

    def choose_file(self):
        file_path = filedialog.askopenfilename(
                    filetypes=[
                        ("Video files", "*.avi *.mp4 *.mov *.mkv *.mpeg *.mpg"),
                        ("All files", "*.*")
                    ]
                )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            self.file_entry.xview_moveto(1)  # Scroll view to the end
    
    def load_rois(self):
        file_path = f"{self.video_path.rsplit('.', 1)[0]}_rois.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                self.loaded_rois = {row["Mouse_ID"] : (row["X"], row["Y"], row["Width"], row["Height"]) for _, row in df.iterrows()}
            except Exception as e:
                    messagebox.showerror("Error", f"Failed to load ROIs: {e}")
        else:
            return
    
    def resume_scoring(self):
        """Allows the user to resume scoring a previously loaded Mouse_ID using a dropdown list."""
        if not self.loaded_rois:
            messagebox.showinfo("Info", "No saved ROIs to resume.")
            return
    
        def on_select():
            """Callback when user clicks 'OK' in the popup."""
            selected_mouse_id = mouse_var.get()
            if selected_mouse_id:
                self.current_mouse_id = selected_mouse_id
                self.rois[selected_mouse_id] = self.loaded_rois[selected_mouse_id]
                self.data[selected_mouse_id] = self.loaded_data.get(selected_mouse_id, {})

                # messagebox.showinfo("Resume Scoring", f"Resumed scoring for Mouse ID: {selected_mouse_id}")
                self.process_frame()
            popup.destroy()
        def on_cancel():
            """Closes the popup without resuming scoring."""
            popup.destroy()
            
        # Create popup window
        popup = tk.Toplevel(self.tk_window)
        popup.title("Resume Scoring")
        popup.geometry("300x150")
        popup.grab_set()  # Make it modal (stay on top)
        
        # Dropdown menu
        mouse_var = tk.StringVar(popup)
        mouse_ids = list(self.loaded_rois.keys())
        mouse_var.set(mouse_ids[-1])  # Default selection
    
        tk.Label(popup, text="Select Mouse ID to resume:").pack(pady=10)
        dropdown = tk.OptionMenu(popup, mouse_var, *mouse_ids)
        dropdown.pack(pady=5)
    
        # Buttons
        button_frame = tk.Frame(popup)
        button_frame.pack(pady=10)
    
        tk.Button(button_frame, text="OK", command=on_select).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
            
    def load_existing_analysis(self):
        self.load_rois()
        if self.loaded_rois:
            for mouse_id, _ in self.loaded_rois.items():
                file_path = f"{self.video_path.rsplit('.', 1)[0]}_{mouse_id}.xlsx"
                if os.path.exists(file_path):
                    # df = pd.read_csv(file_path)
                    df = pd.read_excel(file_path, sheet_name='Immobility')
                    self.loaded_data[mouse_id] = dict(zip(df["Frame"], df["Immobility"]))
    
    # def load_existing_analysis(self):
    #     self.load_rois()
    #     if self.loaded_rois:
    #         for mouse_id, _ in self.loaded_rois.items():
    #             file_path = f"{self.video_path.rsplit('.', 1)[0]}_{mouse_id}.csv"
    #             if os.path.exists(file_path):
    #                 df = pd.read_csv(file_path)
    #                 self.loaded_data[mouse_id] = dict(zip(df["Frame"], df["Immobility"]))
    
    def load_video(self):
        if self.rois:
            confirm = messagebox.askyesno(
                "Warning", "ROIs are not empty. Loading a new video will clear all existing analysis. Continue?")
            if not confirm:
                return
        
        self.frame_number = 0
        self.rois.clear()
        self.data.clear()
        self.current_mouse_id = None
        self.frame_cache.clear()
        
        self.choose_file()
        video_path = self.file_entry.get()
        if not video_path:
            return
        
        self.video_path = video_path
        # self.cap = cv2.VideoCapture(video_path) # different backend if no GStreamer: cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)  # Windows, or cv2.CAP_FFMPEG
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.load_existing_analysis()
        self.process_frame()
    
    def select_roi(self, event=None):
        if self.cap is None:
            return
        
        if self.current_mouse_id and self.current_mouse_id in self.data:
            smallest_frame = min(self.data[self.current_mouse_id].keys(), default=0)
            self.frame_number = smallest_frame  # Set the video to this frame
            if self.auto_save:
                self.save_data()
            else:
                answer = messagebox.askyesno("Save ROI", "Do you want to save the current ROI before selecting a new one?")
                if answer:
                    self.save_data()
            self.jump_to_frame(self.frame_number)
            
        else:
            self.frame_number = 0
            
        ret, frame = self.cap.read()
        if not ret:
            return
        
        height, width = frame.shape[:2]
        scale_factor = 0.5
        frame_resized = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
        roi = cv2.selectROI("Select ROI", frame_resized, fromCenter=False, showCrosshair=True)
        
        if roi != (0, 0, 0, 0):
            mouse_id = tk.simpledialog.askstring("Input", "Enter Mouse ID:")
            if mouse_id:
                roi = tuple(int(element / scale_factor) for element in roi)
                self.rois[mouse_id] = roi
                self.data[mouse_id] = {}
                self.current_mouse_id = mouse_id
                self.update_canvas_with_roi(frame, roi)
        
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        cv2.destroyWindow("Select ROI")
    
    def update_canvas_with_roi(self, frame, roi):
        x, y, w, h = roi
        roi_frame = frame.copy()[y:y + h, x:x + w]
        if len(roi_frame.shape) == 2: #gray
            roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_GRAY2RGB) # not bgr to show in canvas (will be converted anyway)
        
        if self.current_mouse_id in self.data and self.data[self.current_mouse_id]:
            # Get the first frame in which any value (mobility or immobility) is recorded
            first_mobility_frame = min(self.data[self.current_mouse_id].keys())
            time_passed = round((self.frame_number - first_mobility_frame) / self.original_fps,2)
            frame_number_text = f"Frame: {self.frame_number:05d} ({time_passed:.2f}s)"
        else:         
            frame_number_text = f"Frame: {self.frame_number:05d}"
        
        
        cv2.putText(roi_frame, frame_number_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if self.current_mouse_id in self.data: # and self.data[self.current_mouse_id]: #the get_immobility_label can take care of empty data[mouse_id]
            immobility = self.get_immobility_label(self.data[self.current_mouse_id], self.frame_number)
            color = (255,0,0) if 'Immobility' in immobility else (0,255,0)
            cv2.putText(roi_frame, f"{self.current_mouse_id}: {immobility}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        self.roi_frame = roi_frame
        # roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_frame)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        self.canvas.create_image(x, y, anchor=tk.NW, image=tk_img)
        self.canvas.image = tk_img
    
    def get_immobility_label(self, immobility_data, frame_number):
        frame_numbers = immobility_data.keys()
        
        closest_frame = max((f for f in frame_numbers if f <= frame_number), default=None)
        if closest_frame:
            status = 'Immobility' if immobility_data.get(closest_frame) else 'Mobility'
            return f'{status} since {closest_frame}'
        return 'Skip since 0'
       
    def remove_immobility_data(self, event=None):
        """Allows user to remove multiple scoring data entries for specific frames."""
        if self.current_mouse_id not in self.data or not self.data[self.current_mouse_id]:
            messagebox.showwarning("Warning", "No data available for this mouse.")
            return
    
        existing_frames = sorted(self.data[self.current_mouse_id].keys())
    
        # Create a popup window for multiple selection
        selection_window = tk.Toplevel(self.tk_window)
        selection_window.title("Remove Frame Data")
        selection_window.geometry("300x350")
        selection_window.grab_set()  # Keep focus on this window
    
        tk.Label(selection_window, text="Select frame(s) to remove:").pack()
    
        # Listbox with multiple selection enabled
        listbox = tk.Listbox(selection_window, selectmode=tk.MULTIPLE)
        for frame in existing_frames:
            listbox.insert(tk.END, frame)
        listbox.pack(expand=True, fill=tk.BOTH)
    
        def confirm_removal():
            """Delete selected frames when user confirms."""
            selected_frames = [listbox.get(i) for i in listbox.curselection()] 
    
            if not selected_frames:
                messagebox.showwarning("Warning", "No frame selected.")
                return
    
            for frame in selected_frames:
                del self.data[self.current_mouse_id][frame]
    
            messagebox.showinfo("Success", f"Removed frames: {', '.join(map(str, selected_frames))}.")
            selection_window.destroy()
    
        # Confirm button
        tk.Button(selection_window, text="Remove Selected", command=confirm_removal).pack()
    
    def toggle_play(self, event=None):
        self.playing = not self.playing
        self.play_pause_btn.config(text="⏸" if self.playing else "▶")
        self.play_video()
    
    def play_video(self):
        last_time = time.time()
        while self.playing and self.frame_number < self.total_frames-1:
            current_time = time.time()
            elapsed_time = current_time - last_time
            
            if elapsed_time >= 1 / self.play_fps:
                self.process_frame()
                self.update_time_display()
                last_time = current_time
                self.tk_window.update_idletasks()
                self.tk_window.update()
                # Only increase frame_number *if still playing*
                if self.playing:
                    self.frame_number += 1
     
    def next_frame(self, event=None):
        if self.cap and self.frame_number < self.total_frames-1 and not self.playing:
            self.frame_number += 1
            self.process_frame()
     
    def prev_frame(self, event=None):
        if self.cap and self.frame_number > 0 and not self.playing:
            self.frame_number -= 1
            # self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            if self.frame_number in [f[0] for f in self.frame_cache]:
                self.process_frame()
            else:
                # self.frame_number += 1
                # or allow to navigate
                self.jump_to_frame(self.frame_number)
            
    def mark_immobility(self, value):
        if self.current_mouse_id is not None:
            self.data[self.current_mouse_id][self.frame_number] = value
        
        # change button color
        if value == 0:
            self.immobility_button.config(background="black") #SystemButtonFace
            self.mobility_button.config(background="green")
        else:
            self.immobility_button.config(background="red")
            self.mobility_button.config(background="black")
        
        if not self.playing:
            self.toggle_play()
            
    def on_number_key(self, event, value):
        # if not file_entry.focus_get(): # not entering value like mouse ID
        if event.widget not in [self.file_entry, self.speed_entry,self.frame_entry]:
            self.mark_immobility(value)
        
    def save_and_next_mouse(self):
        self.auto_save = True
        self.select_roi()
        self.auto_save = False

    def is_file_open(self,filepath):
        """Check if a file is open in another program."""
        try:
            with open(filepath, "r+"):
                return False  # File is accessible
        except IOError:
            return True  # File is open elsewhere
    
    def save_data(self):
        if not self.data or self.current_mouse_id not in self.data:
            messagebox.showwarning("Warning", "No data to save!")
            return
    
        base_filepath = f"{self.video_path.rsplit('.', 1)[0]}_{self.current_mouse_id}.xlsx"
    
        if os.path.exists(base_filepath):
            if self.is_file_open(base_filepath):
                messagebox.showerror("Error", f"{base_filepath} is open in another program. Please close it first.")
                return
            
            response = messagebox.askyesnocancel("File Exists", f"{base_filepath} already exists. Overwrite?")
    
            if response is None:
                return  # User canceled the action
    
            if not response:  # User chose NOT to overwrite
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_filepath = f"{self.video_path.rsplit('.', 1)[0]}_{self.current_mouse_id}_backup_{timestamp}.xlsx"
                shutil.move(base_filepath, backup_filepath)
                messagebox.showinfo("Backup Created", f"Existing file renamed to {backup_filepath}")
    
        # Save new data with the original filename
        df = pd.DataFrame(sorted(self.data[self.current_mouse_id].items()), columns=["Frame", "Immobility"]) #sort by frame and return a list
        df_summary = self.fill_frame_time(df)
        #add cum. for all frames
        df_summary['cum(s)'] = round(df_summary["Immobility"].cumsum() /self.original_fps, 2)
        time_column = 'Time(s)'
        df_summary['TrialTime'] = df_summary[time_column] - df_summary[time_column].iloc[0]
        time_step = df_summary['TrialTime'].diff().mean().item()
        df_summary['AbsFrameTime'] = (df_summary['Frame'] -1) * time_step
        
        df_bined = self.bined_immobility(df_summary,10)
        df_bined_30 = self.bined_immobility(df_summary,30)
        
        self.save_to_excel_one(base_filepath,dfs=[df,df_summary,df_bined, df_bined_30], sheet_names = ['Immobility','Summary','dfs_bin_time_df', 'dfs_bin30_time_df'])
        
        # df.to_csv(base_filepath, index=False)
        messagebox.showinfo("Save", f"Data saved successfully as {base_filepath}!")
    
        self.save_rois_to_csv()  # Can be overwritten
    
    def save_to_excel_one(self, base_filepath, dfs, sheet_names):
        if dfs: #the tosave_list is not empty

            def excel_writer():
                wb = Workbook()
                for i,df in enumerate(dfs):
                    if not df.empty:
                        data = [df.columns.tolist(),] + df.values.tolist()
                        wb.new_sheet(sheet_name =sheet_names[i], data=data)
                wb.save(base_filepath)
            
            excel_writer()
    
    def save_rois_to_csv(self):
        filepath = f"{self.video_path.rsplit('.', 1)[0]}_rois.csv"
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            existing_mouse_ids = set(existing_df["Mouse_ID"])
        else:
            existing_mouse_ids = set()

        # Filter out ROIs that already exist in the CSV
        new_rois = {mouse_id: roi for mouse_id, roi in self.rois.items() if mouse_id not in existing_mouse_ids}

        if new_rois:
            new_df = pd.DataFrame(
                [(mouse_id, *roi) for mouse_id, roi in new_rois.items()],
                columns=["Mouse_ID", "X", "Y", "Width", "Height"]
            )
            new_df.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)
                    
    def compress_frame_webp(self,gray_frame, quality=60):
        pil_image = Image.fromarray(gray_frame)
        buf = io.BytesIO()
        pil_image.save(buf, format='WebP', quality=quality,method=0) #fastest compression method; 6 is best but slowest
        return buf.getvalue()

    def decompress_frame_webp(self,webp_bytes):
        buf = io.BytesIO(webp_bytes)
        pil_image = Image.open(buf)
        return np.array(pil_image)
    
    def compress_frame_jpg(self,gray_frame, quality=80):
        if not isinstance(gray_frame, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if gray_frame.ndim != 2:
            raise ValueError("Expected a grayscale (2D) image.")
            
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded_img = cv2.imencode('.jpg', gray_frame, encode_param)
        if not success:
            raise ValueError("JPEG compression failed.")
        return encoded_img.tobytes()
    
    def decompress_frame_jpg(self,jpg_bytes):
        jpg_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(jpg_array, cv2.IMREAD_GRAYSCALE)
        return frame
    
    def process_frame(self): # try compression before adding to cache
        if self.cap is None:
            return
        if self.frame_number in [f[0] for f in self.frame_cache]:
            frame = next(f[1] for f in self.frame_cache if f[0] == self.frame_number)
            frame = self.decompress_frame_jpg(frame)
            #debug
            if self.debug:
                print(f'Current frame: {self.frame_number}')
        
        else:
            ret, frame = self.cap.read()
            if not ret:
                return
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # self.frame_cache.append((self.frame_number, frame.copy()))
            jpg_bytes = self.compress_frame_jpg(frame, quality=80)
            self.frame_cache.append((self.frame_number, jpg_bytes))
            # print("Original frame size:", frame.nbytes, "bytes")
            # print("Compressed jpg size:", len(jpg_bytes), "bytes")
            #debug
            if self.debug:
                print(f'Current frame: {self.cap.get(cv2.CAP_PROP_POS_FRAMES)-1}')
        if self.rois:
            self.update_canvas_with_roi(frame, self.rois[self.current_mouse_id])
            self.update_time_display()
        else:
            # load the rois file for the show (after analysis)
            if self.loaded_rois:            #need to load data as well
                for mouse_id, (x, y, w, h) in self.loaded_rois.items():
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if mouse_id in self.loaded_data.keys():
                        immobility = self.get_immobility_label(self.loaded_data[mouse_id], self.frame_number)
                    else:
                        immobility = ""
                    cv2.putText(frame, f"{mouse_id}: {immobility}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            height, width = frame.shape[:2]
            roi = (0, 0, width, height)
            self.canvas.config(width=width, height=height)
            self.update_canvas_with_roi(frame, roi)
    
    def update_time_display(self):        
        if self.current_mouse_id in self.data and self.data[self.current_mouse_id]:
            # Get the first frame in which any value (mobility or immobility) is recorded
            first_mobility_frame = min(self.data[self.current_mouse_id].keys())
            last_event_frame = max((f for f in self.data[self.current_mouse_id].keys() if f <= self.frame_number), default=0)
            time_passed = round((self.frame_number - first_mobility_frame) / self.original_fps,2)
            time_escaped = round((self.frame_number - last_event_frame) / self.original_fps,2)
            # minutes_pass = int(time_passed // 60)
            # seconds_pass = time_passed % 60
            # minutes_escape = int(time_escaped // 60)
            # seconds_escape = time_escaped % 60
            self.timer_label.config(text=f"Time since {first_mobility_frame}: {time_passed:06.2f}, Time since last: {time_escaped:06.2f}")
            
    def select_cam_file(self):
        folder_path = os.path.dirname(self.video_path)
        default_filename = None
        
        # Parse video path to get base name and timestamp
        match = re.match(r"(.*)_([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}_[0-9]{2}_[0-9]{2})\.(avi|mp4|mov|mkv)", self.video_path,re.IGNORECASE)
        if match:
            base_name, timestamp,_ = match.groups() # basename, timestamp,ext
            default_filename = f"{base_name}_cam_{timestamp}.csv"
        
        if default_filename and os.path.isfile(default_filename):
            return default_filename
           
        else:# Open file dialog
            root = tk.Tk()
            root.withdraw()
        
            file_path = filedialog.askopenfilename(
                title="Select frame-time CSV file",
                initialdir=folder_path,
                initialfile="",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
        
            root.destroy()
        
            return file_path if file_path else None
    
    def fill_frame_time(self, df_data):
        start_frame = df_data['Frame'].iloc[0] #min(self.data[self.current_mouse_id].keys())
        time_column = 'Time(s)'
        cam_filename = self.select_cam_file()
        if cam_filename and os.path.isfile(cam_filename):
            df_raw = pd.read_csv(cam_filename, header=0)
            # Filter data based on Col3 values
            if 'systemTime' in df_raw.columns: # neurophotometrics time at sec
                df = df_raw[['systemTime', 'Cam_Frame']].rename(columns={'systemTime': time_column, 'Cam_Frame': 'Frame'})
                print('Merge with systemTimeStamp')
            else: # computer time at ms
                df = df_raw[['Millis', 'Cam_Frame']].rename(columns={'Millis': time_column, 'Cam_Frame': 'Frame'})
                df[time_column] = df[time_column]/1000
                print('Merge with computerTimeStamp')

            
            df['Frame'] = df['Frame'] - df['Frame'].iloc[0] + 1 # in case the frame does not start from 1
            start_time = df.loc[df["Frame"] == start_frame, time_column].iloc[0]  # Get Col2 at first Col3=100
            end_time = start_time + self.session_duration
            df = df[(df[time_column] >= start_time) & (df[time_column] < end_time)]

            # Reset index
            df = df.reset_index(drop=True)
            
        else:
            # if no file, build a df
            print(f'build the time for summary based on original fps: {self.original_fps}')
            times = np.arange(0, self.session_duration, 1 / self.original_fps)
            times = np.round(times, 2)  # optional: round to 2 decimal places
            frames = np.arange(start_frame, start_frame + len(times))

            df = pd.DataFrame({
                time_column: times,
                'Frame': frames
                })
        #fill the 3nd col with immobility
        df = self.add_immobility_col(df,df_data)    
        return df

    def fill_frame_time_old(self, df_data):
        start_frame = df_data['Frame'].iloc[0] #min(self.data[self.current_mouse_id].keys())
        time_column = 'Time(s)'
        cam_filename = self.select_cam_file()
        if cam_filename and os.path.isfile(cam_filename):
            df = pd.read_csv(cam_filename, header=0, usecols=[1, 2])
            # Filter data based on Col3 values
            
            df.columns = [time_column, "Frame"]
            df[time_column] = df[time_column]/1000
            df['Frame'] = df['Frame'] - df['Frame'].iloc[0] + 1 # in case the frame does not start from 1
            start_time = df.loc[df["Frame"] == start_frame, time_column].iloc[0]  # Get Col2 at first Col3=100
            end_time = start_time + self.session_duration
            df = df[(df[time_column] >= start_time) & (df[time_column] < end_time)]

            # Reset index
            df = df.reset_index(drop=True)
            
        else:
            # if no file, build a df
            print(f'build the time for summary based on original fps: {self.original_fps}')
            times = np.arange(0, self.session_duration, 1 / self.original_fps)
            times = np.round(times, 2)  # optional: round to 2 decimal places
            frames = np.arange(start_frame, start_frame + len(times))

            df = pd.DataFrame({
                time_column: times,
                'Frame': frames
                })
        #fill the 3nd col with immobility
        df = self.add_immobility_col(df,df_data)    
        return df
    
    def add_immobility_col(self,df, df_DIO): #df_DIO could be DIO_indices or DIO file from bonsai, not only 0/1
        # Initialize the value column in A with zeros
        align_column = df_DIO.columns[0]
        DIOs = df_DIO.columns[1:] #DIOs = df_DIO.columns.difference([time_column])
        
        df[DIOs] = df_DIO[DIOs].iloc[-1] # make sure the end
    
        # Vectorized approach to update the value column based on conditions from df_DIO
        for i in range(len(df_DIO) - 1):
            start_frame = df_DIO.at[i, align_column]  # Use df.at for fast access to a single scalar value
            end_frame = df_DIO.at[i + 1, align_column]
            
            # Create a mask for the time range
            mask = (df[align_column] >= start_frame) & (df[align_column] < end_frame)
            
            # Assign values for each DIO column within the mask
            df.loc[mask, DIOs] = df_DIO.loc[i, DIOs].values  
        
        return df
    
    def bined_immobility(self,df_summary, bin_size): 
        ''' calculate bin-timed immobility'''
        # Step 1: Create bins based on time_column and bin_size
        time_column = 'Time(s)'
        df=df_summary.copy()
        
        df['bin'] = ((df[time_column]-df[time_column].iloc[0]) // bin_size) * bin_size
        
        # Step 2: Calculate total mobility time in each bin using sum
        immobility_time_per_bin = df.groupby('bin')['Immobility'].sum() * bin_size / df.groupby('bin')['Immobility'].count()
        
        
        # Step 3: Combine the results into a DataFrame
        bin_times_df = pd.DataFrame({
            'bin': immobility_time_per_bin.index,  # The start of each bin
            'immobility_time': immobility_time_per_bin.values
        })
        
        bin_times_df["cum_time"] = bin_times_df["immobility_time"].cumsum()
        return bin_times_df
    
    def run(self):
        self.tk_window.mainloop()

if __name__ == "__main__":
    app = VideoScoring()
    app.debug = 0
    app.run()
