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

class VideoScoring:
    def __init__(self):
        self.version = 'zy_iMOSS_MV_20251119_withDuration.py'
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
            img = Image.open('iMOSS-V_new.png')
            img = img.resize((200,40),Image.LANCZOS)
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
        tk.Button(third_frame, text="Save Data", command=self.save_data, bg="purple", fg="white", font = bold_font,width=20).grid(row=0, column=5, sticky=tk.E, padx=5)
        

        self.duration_entry = tk.Entry(third_frame, width=9)
        self.duration_entry.grid(row=0, column=3, sticky=tk.E, padx=0)
        self.duration_entry.insert(0, self.session_duration)
        
        tk.Button(third_frame, text="Set_Dur", command=self.set_duration, font = bold_font,width=6).grid(row=0, column=4, sticky=tk.E, padx=5)
        # self.duration_label = tk.Label(third_frame, text="Duration", font = bold_font, fg="purple", width=6)
        # self.duration_label.grid(row=0, column=3, sticky=tk.E, padx=0)
        

        
        # spacer.grid(row=0, column=4,padx=5)
        
        self.immobility_button = tk.Button(third_frame, text="Immobility (1)", command=lambda: self.mark_immobility(1), bg = 'black', fg = 'white',font = bold_font, width=20)
        self.immobility_button.grid(row=0, column=6, sticky=tk.E, padx=5)
       
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
    
    def set_duration(self):
        # self.tk_window.focus()
        try:
            self.session_duration = max(0, int(self.duration_entry.get()))
        except ValueError:
            print('Use 360 s')
            self.session_duration = 360

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
        
        # --- Auto scale for display ---       
        # win_w = self.tk_window.winfo_width()
        # win_h = self.tk_window.winfo_height()
        # print("window size:", win_w, win_h)
        win_w = self.canvas.winfo_width()
        win_h = self.canvas.winfo_height()
    
        img_h, img_w = roi_frame.shape[:2]
    
        scale = min(win_w / img_w, win_h / img_h)
        new_size = (int(img_w * scale), int(img_h * scale))
        roi_frame = cv2.resize(roi_frame, new_size, interpolation=cv2.INTER_AREA)
            
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
                       # add note
            immobility = self.get_immobility_label(self.data[self.current_mouse_id], self.frame_number)
            color = (255,0,0) if 'Immobility' in immobility else (0,255,0)
            cv2.putText(roi_frame, f"{self.current_mouse_id}: {immobility}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)      
      
        self.roi_frame = roi_frame
        # roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_frame)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
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
            else: # computer time at ms
                df = df_raw[['Millis', 'Cam_Frame']].rename(columns={'Millis': time_column, 'Cam_Frame': 'Frame'})
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
