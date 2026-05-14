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
20260501: fix potential tk-cv2 confliction in new Spyder (not finished)
@author: Zengyou Ye at NIH/NIDA/IRP (addy9908@gmail.com)
"""

# %gui tk  # Uncomment this line ONLY if running inside Spyder's interactive console!

import time, os
import re
import shutil
import cv2
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, Canvas, messagebox, simpledialog
from tkinter import ttk
from collections import deque
from PIL import Image, ImageTk
from pyexcelerate import Workbook
import warnings

warnings.simplefilter("ignore", UserWarning)

class VideoScoring:
    def __init__(self):
        # --- App Info & Settings ---
        self.version = 'zy_iMOSS_MV_20260505_Verified'
        self.play_fps = 30
        self.session_duration = 360
        self.debug = 0
        self.auto_save = False
        self.help_visible = False
        
        # --- Event Naming (Bug Fix 4) ---
        self.event_names = {0: "Mobility", 1: "Immobility"}

        # --- Video & State Management ---
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.original_fps = 30
        self.frame_number = 0
        self.playing = False
        self.frame_cache = deque(maxlen=28000)

        # --- Data & ROI Management ---
        self.data = {}          
        self.rois = {}          
        self.loaded_rois = {}   
        self.loaded_data = {}   
        self.current_mouse_id = None
        
        self.roi_drawing_mode = False 
        self.roi_rect_id = None
        self.roi_start_x = 0
        self.roi_start_y = 0

        # --- UI Initialization ---
        self.tk_window = tk.Tk()
        self.init_ui()

    def init_ui(self):
        """Builds the main Tkinter UI using a professional grid layout."""
        self.tk_window.title("iMOSS-MV: Immobility & Mobility Optimized Scoring System")
        self.tk_window.geometry("1400x800") 
        self.tk_window.configure(bg='#e0e0e0')

        # --- Main Layout (Bug Fix 4: Scoring Left, Video Right) ---
        self.tk_window.grid_rowconfigure(0, weight=1)
        self.tk_window.grid_columnconfigure(0, weight=1) # Scoring panel takes 1/4 
        self.tk_window.grid_columnconfigure(1, weight=4) # Video panel takes 3/4

        # ==========================================
        # LEFT PANEL (Scoring & Stats)
        # ==========================================
        left_panel = ttk.Frame(self.tk_window, padding=10)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left_panel.grid_rowconfigure(2, weight=1) 
        left_panel.grid_columnconfigure(0, weight=1)

        ttk.Label(left_panel, text="Scoring Panel", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky='w')
        ttk.Separator(left_panel, orient='horizontal').grid(row=1, column=0, columnspan=2, sticky='ew', pady=10)

        # --- Scoring Table (Treeview) ---
        tree_frame = ttk.Frame(left_panel)
        tree_frame.grid(row=2, column=0, columnspan=2, sticky='nsew')
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        self.scoring_tree = ttk.Treeview(tree_frame, columns=("Frame", "Score"), show="headings", selectmode="extended")
        self.scoring_tree.heading("Frame", text="Frame")
        self.scoring_tree.heading("Score", text="Score")
        self.scoring_tree.column("Frame", width=100, anchor='center')
        self.scoring_tree.column("Score", width=100, anchor='center')
        
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.scoring_tree.yview)
        self.scoring_tree.configure(yscrollcommand=scrollbar.set)
        
        self.scoring_tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')

        tk.Button(left_panel, text="Remove Selected Score(s) (Del)", command=self.remove_selected_scores, bg="#ff4d4d", fg="white", font=('Helvetica', 9, 'bold')).grid(row=3, column=0, columnspan=2, pady=5, sticky='ew')

        # --- Live Immobility Counter ---
        stats_frame = ttk.Frame(left_panel, padding=(0, 10))
        stats_frame.grid(row=4, column=0, columnspan=2, sticky='ew')
        
        ttk.Label(stats_frame, text="Total Immobility (1) Time:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky='w')
        self.total_immobility_label = ttk.Label(stats_frame, text="0.00 s", font=("Segoe UI", 10, "bold"), foreground="blue")
        self.total_immobility_label.grid(row=0, column=1, sticky='w', padx=5)

        ttk.Label(stats_frame, text="Current Bout (1) Duration:", font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky='w', pady=(5,0))
        self.current_immobility_label = ttk.Label(stats_frame, text="0.00 s", font=("Segoe UI", 10, "bold"), foreground="purple")
        self.current_immobility_label.grid(row=1, column=1, sticky='w', padx=5, pady=(5,0))

        # --- Utility Buttons ---
        action_frame = ttk.Frame(left_panel)
        action_frame.grid(row=5, column=0, columnspan=2, sticky='ew', pady=(10, 0))
        action_frame.grid_columnconfigure((0,1), weight=1)
        
        tk.Button(action_frame, text="Draw New ROI (r)", command=self.start_roi_drawing, bg="orange", fg="black", font=('Helvetica', 9, 'bold')).grid(row=0, column=0, sticky='ew', padx=(0, 5))
        tk.Button(action_frame, text="Resume Scoring", command=self.resume_scoring, bg="orange", fg="black", font=('Helvetica', 9, 'bold')).grid(row=0, column=1, sticky='ew', padx=(5, 0))
        tk.Button(action_frame, text="Save ROI as PNG (600dpi)", command=self.save_roi_as_png, bg="#008CBA", fg="white", font=('Helvetica', 9, 'bold')).grid(row=1, column=0, columnspan=2, sticky='ew', pady=(10, 0))
          
        tk.Button(left_panel, text="Save Data", command=self.save_data, bg="purple", fg="white", font=('Helvetica', 10, 'bold')).grid(row=6, column=0, columnspan=2, pady=(15,5), sticky='ew')
        tk.Button(left_panel, text="Save & Next Mouse", command=self.save_and_next_mouse, bg="purple", fg="white", font=('Helvetica', 10, 'bold')).grid(row=7, column=0, columnspan=2, pady=5, sticky='ew')
                
        # ==========================================
        # RIGHT PANEL (Video & Controls)
        # ==========================================
        right_panel = tk.Frame(self.tk_window, bg='#e0e0e0')
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        # --- Top Controls ---
        controls_frame = tk.Frame(right_panel, bg='#e0e0e0')
        controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        tk.Button(controls_frame, text="Load Video", command=self.load_video, bg="red", fg="white", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT)
        self.file_entry = ttk.Entry(controls_frame, width=60, justify='right')
        self.file_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
        self.timer_label = ttk.Label(controls_frame, text="Time (s): N/A", font=("Segoe UI", 10, "bold"), foreground="red")
        self.timer_label.pack(side=tk.LEFT)
        
        # --- Main Video Canvas ---
        self.canvas = Canvas(right_panel, bg="black", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self.on_roi_start)
        self.canvas.bind("<B1-Motion>", self.on_roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_roi_end)

        # --- Playback Controls ---
        playback_frame = tk.Frame(right_panel, bg='#e0e0e0')
        playback_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        
        tk.Button(playback_frame, text="⏮ Prev", command=self.prev_frame, font=('Helvetica', 10, 'bold'), width=10).pack(side=tk.LEFT, padx=2)
        self.play_pause_btn = tk.Button(playback_frame, text="▶ Play", command=self.toggle_play, bg="blue", fg="white", font=('Helvetica', 10, 'bold'), width=10)
        self.play_pause_btn.pack(side=tk.LEFT, padx=2)
        tk.Button(playback_frame, text="Next ⏭", command=self.next_frame, font=('Helvetica', 10, 'bold'), width=10).pack(side=tk.LEFT, padx=2)
        
        self.frame_entry = ttk.Entry(playback_frame, width=10)
        self.frame_entry.pack(side=tk.LEFT, padx=(10, 2))
        self.frame_entry.insert(0, 0)
        tk.Button(playback_frame, text="Jump", command=lambda: self.jump_to_frame(int(self.frame_entry.get())), font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT, padx=2)
        
        self.mobility_button = tk.Button(playback_frame, text="Mobility (0)", command=lambda: self.mark_immobility(0), bg='black', fg='white', font=('Helvetica', 10, 'bold'))
        self.mobility_button.pack(side=tk.RIGHT, padx=5)
        self.immobility_button = tk.Button(playback_frame, text="Immobility (1)", command=lambda: self.mark_immobility(1), bg='black', fg='white', font=('Helvetica', 10, 'bold'))
        self.immobility_button.pack(side=tk.RIGHT, padx=5)

        # --- Settings Frame ---
        settings_frame = ttk.Frame(right_panel)
        settings_frame.grid(row=3, column=0, sticky='e', pady=(5, 0))
        ttk.Label(settings_frame, text="FPS:").pack(side=tk.LEFT)
        self.speed_entry = ttk.Entry(settings_frame, width=5)
        self.speed_entry.pack(side=tk.LEFT, padx=5)
        self.speed_entry.insert(0, self.play_fps)
        ttk.Button(settings_frame, text="Set", command=self.set_play_fps, width=5).pack(side=tk.LEFT)
        
        ttk.Label(settings_frame, text="Dur (s):").pack(side=tk.LEFT, padx=(10,0))
        self.duration_entry = ttk.Entry(settings_frame, width=5)
        self.duration_entry.pack(side=tk.LEFT, padx=5)
        self.duration_entry.insert(0, self.session_duration)
        ttk.Button(settings_frame, text="Set", command=self.set_duration, width=5).pack(side=tk.LEFT)
        ttk.Button(settings_frame, text="Shortcut List (H)", command=self.toggle_help, width=15).pack(side=tk.LEFT, padx=(10,0))

        # --- Key Bindings ---
        self.tk_window.bind("<KeyPress-space>", self.toggle_play)
        self.tk_window.bind("<KeyPress-Left>", self.prev_frame)
        self.tk_window.bind("<KeyPress-Right>", self.next_frame)
        self.tk_window.bind("<KeyPress-r>", self.start_roi_drawing)
        self.tk_window.bind("<KeyPress-0>", lambda e: self.on_number_key(e, 0))
        self.tk_window.bind("<KeyPress-1>", lambda e: self.on_number_key(e, 1))
        self.tk_window.bind("<KeyPress-Delete>", self.remove_selected_scores)
        self.tk_window.bind("<KeyPress-h>", self.toggle_help)
        
        self.tk_window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        def unfocus_if_not_entry(event):
            widget = event.widget
            if not isinstance(widget, tk.Entry):
                self.tk_window.focus_set()
                
        self.tk_window.bind("<Button-1>", unfocus_if_not_entry) # move cursor out of entries

    def restore_focus(self):
        """Fixes Bug 1: Prevents spacebar from reactivating buttons."""
        self.tk_window.focus_set()

    # ==========================================
    # FILE LOADING & SETUP
    # ==========================================
    def choose_file(self):
        file_path = filedialog.askopenfilename(
            parent=self.tk_window,
            filetypes=[("Video files", "*.avi *.mp4 *.mov *.mkv *.mpeg *.mpg"), ("All files", "*.*")]
        )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def load_video(self):
        self.restore_focus()
        if self.rois:
            confirm = messagebox.askyesno("Warning", "ROIs are not empty. Loading a new video will clear all existing analysis. Continue?", parent=self.tk_window)
            if not confirm: return
        
        self.frame_number = 0
        self.rois.clear()
        self.data.clear()
        self.current_mouse_id = None
        self.frame_cache.clear()
        self.update_scoring_table()
        
        self.choose_file()
        video_path = self.file_entry.get()
        if not video_path: return
        
        self.video_path = video_path
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            

        self.cap = cv2.VideoCapture(video_path) 
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.", parent=self.tk_window)
            return
            
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.original_fps == 0: self.original_fps = 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Bug Fix 3: Automatically load existing data when a video is loaded
        self.load_existing_analysis()
        self.process_frame()

    # ==========================================
    # NATIVE TKINTER ROI DRAWING
    # ==========================================
    def start_roi_drawing(self, event=None):
        self.restore_focus()
        if self.cap is None: return
        if self.playing: self.toggle_play()

        if self.current_mouse_id and self.data.get(self.current_mouse_id):
            response = messagebox.askyesnocancel(
                "Save Current Analysis?",
                f"Do you want to save the data for '{self.current_mouse_id}' before starting a new ROI?",
                parent=self.tk_window
            )
            if response is True:  
                self.save_data()
            elif response is None:  
                return  

        self.roi_drawing_mode = True
        self.current_mouse_id = None # Forces full-frame view to draw new ROI
        self.canvas.config(cursor="cross")
        self.process_frame() 

    def on_roi_start(self, event):
        if not self.roi_drawing_mode: return
        self.roi_start_x = event.x
        self.roi_start_y = event.y
        if self.roi_rect_id:
            self.canvas.delete(self.roi_rect_id)
        self.roi_rect_id = self.canvas.create_rectangle(self.roi_start_x, self.roi_start_y, self.roi_start_x, self.roi_start_y, outline='red', width=2)

    def on_roi_drag(self, event):
        if not self.roi_drawing_mode or not self.roi_rect_id: return
        self.canvas.coords(self.roi_rect_id, self.roi_start_x, self.roi_start_y, event.x, event.y)

    def on_roi_end(self, event):
        if not self.roi_drawing_mode: return
        self.roi_drawing_mode = False
        self.canvas.config(cursor="")
        
        # if self.roi_rect_id: 
        #     self.canvas.delete(self.roi_rect_id)
        #     self.roi_rect_id = None

        if abs(self.roi_start_x - event.x) < 5 or abs(self.roi_start_y - event.y) < 5:
            self.process_frame(); return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        ret, frame = self.cap.read()
        if not ret: return
        
        orig_h, orig_w = frame.shape[:2]
        scale = min(canvas_w / orig_w, canvas_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (canvas_w - new_w) // 2
        pad_y = (canvas_h - new_h) // 2
        
        orig_x = int(max(0, min(self.roi_start_x, event.x) - pad_x) / scale)
        orig_y = int(max(0, min(self.roi_start_y, event.y) - pad_y) / scale)
        orig_rect_w = int(abs(event.x - self.roi_start_x) / scale)
        orig_rect_h = int(abs(event.y - self.roi_start_y) / scale)
        
        roi = (orig_x, orig_y, orig_rect_w, orig_rect_h)
        
        mouse_id = simpledialog.askstring("Input", "Enter Mouse ID:", parent=self.tk_window)
        if mouse_id:
            self.rois[mouse_id] = roi
            self.data[mouse_id] = {}
            self.current_mouse_id = mouse_id
            self.process_frame()
            self.update_scoring_table()
            self.update_immobility_stats()
        else:
            self.process_frame() 

    # ==========================================
    # VIDEO PROCESSING & DISPLAY
    # ==========================================
    def process_frame(self):
        if self.cap is None or not self.cap.isOpened(): return

        # Check Cache
        if self.frame_number in [f[0] for f in self.frame_cache]:
            frame = next(f[1] for f in self.frame_cache if f[0] == self.frame_number)
            frame = self.decompress_frame_jpg(frame)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            ret, frame = self.cap.read()
            if not ret: return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            jpg_bytes = self.compress_frame_jpg(frame, quality=80)
            self.frame_cache.append((self.frame_number, jpg_bytes))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Bug Fix 3: Show Full Frame with ALL loaded ROIs if no mouse is actively selected
        if not self.roi_drawing_mode and self.rois and self.current_mouse_id in self.rois:
            x, y, w, h = self.rois[self.current_mouse_id]
            display_frame = frame_rgb[y:y+h, x:x+w].copy()
        else:
            display_frame = frame_rgb.copy()
            # Draw loaded/existing ROIs on the full frame
            # all_rois_to_draw = {**self.loaded_rois, **self.rois}
            for m_id, (rx, ry, rw, rh) in self.rois.items():
                cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{m_id}", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.add_overlays(display_frame)
        self.update_canvas_image(display_frame)
        self.update_time_display() 

    def add_overlays(self, display_frame):
        frame_text = f"Frame: {self.frame_number:05d}/{self.total_frames}"
        cv2.putText(display_frame, frame_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
        
        if self.current_mouse_id in self.data:
            immobility = self.get_immobility_label(self.data[self.current_mouse_id], self.frame_number)
            color = (255, 0, 0) if immobility == self.event_names[1] else (0, 255, 0) 
            cv2.putText(display_frame, f"{self.current_mouse_id}: {immobility}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        self.roi_frame = display_frame.copy()

    def update_canvas_image(self, display_frame):
        win_w = self.canvas.winfo_width()
        win_h = self.canvas.winfo_height()
        if win_w < 10: win_w, win_h = 800, 600 
    
        img_h, img_w = display_frame.shape[:2]
        scale = min(win_w / img_w, win_h / img_h)
        new_size = (int(img_w * scale), int(img_h * scale))
        
        resized_frame = cv2.resize(display_frame, new_size, interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized_frame)
        self.tk_img = ImageTk.PhotoImage(image=pil_img)
        
        pad_x = (win_w - new_size[0]) // 2
        pad_y = (win_h - new_size[1]) // 2
        
        self.canvas.delete("all")
        self.canvas.create_image(pad_x, pad_y, anchor=tk.NW, image=self.tk_img)

    def compress_frame_jpg(self, gray_frame, quality=80):
        success, encoded_img = cv2.imencode('.jpg', gray_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return encoded_img.tobytes()
    
    def decompress_frame_jpg(self, jpg_bytes):
        return cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    # ==========================================
    # PLAYBACK CONTROLS
    # ==========================================
    def toggle_play(self, event=None):
        if not self.cap: return
        if event and isinstance(event.widget, (tk.Entry, ttk.Entry)): return
        
        self.playing = not self.playing
        self.play_pause_btn.config(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing: self.play_video()
    
    def play_video(self):
        last_time = time.time()
        while self.playing and self.frame_number < self.total_frames-1:
            current_time = time.time()
            if current_time - last_time >= 1 / self.play_fps:
                self.process_frame()
                last_time = current_time
                self.tk_window.update_idletasks()
                self.tk_window.update()
                if self.playing:
                    self.frame_number += 1
                    self.update_immobility_stats()

    def next_frame(self, event=None):
        self.restore_focus()
        if self.cap and self.frame_number < self.total_frames-1 and not self.playing:
            self.frame_number += 1
            self.process_frame()
     
    def prev_frame(self, event=None):
        self.restore_focus()
        if self.cap and self.frame_number > 0 and not self.playing:
            self.frame_number -= 1
            self.process_frame()

    def jump_to_frame(self, target_frame):
        self.restore_focus()
        if 0 <= target_frame < self.total_frames:
            self.frame_number = target_frame
            self.process_frame()

    def set_play_fps(self):
        self.restore_focus()
        try: self.play_fps = max(1, int(self.speed_entry.get()))
        except ValueError: self.play_fps = self.original_fps
    
    def set_duration(self):
        self.restore_focus()
        try: self.session_duration = max(0, int(self.duration_entry.get()))
        except ValueError: self.session_duration = 360

    # ==========================================
    # DATA & SCORING LOGIC
    # ==========================================
    def on_number_key(self, event, value):
        if not isinstance(event.widget, (tk.Entry, ttk.Entry)):
            self.mark_immobility(value)

    def mark_immobility(self, value):
        if self.current_mouse_id is not None:
            self.data[self.current_mouse_id][self.frame_number] = value
            self.update_scoring_table()
            self.update_immobility_stats()
            
        if value == 0:
            self.immobility_button.config(bg="black")
            self.mobility_button.config(bg="green")
        else:
            self.immobility_button.config(bg="red")
            self.mobility_button.config(bg="black")
        
        if not self.playing:
            self.toggle_play()

    # def get_immobility_label(self, immobility_data, frame_number):
    #     if not immobility_data: return 'Not Scored'
    #     frame_numbers = immobility_data.keys()
    #     closest_frame = max((f for f in frame_numbers if f <= frame_number), default=None)
    #     if closest_frame is not None:
    #         score = immobility_data.get(closest_frame)
    #         return self.event_names.get(score, "Unknown")
    #     return 'Not Scored'
    def get_immobility_label(self, immobility_data, frame_number):
        frame_numbers = immobility_data.keys()
        
        closest_frame = max((f for f in frame_numbers if f <= frame_number), default=None)
        if closest_frame:
            status = 'Immobility' if immobility_data.get(closest_frame) else 'Mobility'
            return f'{status} since {closest_frame}'
        return 'Skip since 0'
    
    def update_scoring_table(self):
        for item in self.scoring_tree.get_children():
            self.scoring_tree.delete(item)
            
        if self.current_mouse_id and self.current_mouse_id in self.data:
            sorted_frames = sorted(self.data[self.current_mouse_id].items())
            for frame_num, score in sorted_frames:
                score_name = self.event_names.get(score, "Unknown")
                self.scoring_tree.insert("", tk.END, iid=frame_num, values=(frame_num, score_name))
            
            # Auto-scroll to bottom
            if sorted_frames:
                last_item = self.scoring_tree.get_children()[-1]
                self.scoring_tree.selection_set(last_item)
                self.scoring_tree.see(last_item)

    def remove_selected_scores(self, event=None):
        self.restore_focus()
        selected_items = self.scoring_tree.selection()
        if not selected_items: return
        
        if self.current_mouse_id and self.current_mouse_id in self.data:
            for item_id in selected_items:
                frame_num = int(item_id)
                if frame_num in self.data[self.current_mouse_id]:
                    del self.data[self.current_mouse_id][frame_num]
            
            self.update_scoring_table()
            self.update_immobility_stats()
            self.process_frame()

    def update_time_display(self):        
        if self.current_mouse_id and self.data.get(self.current_mouse_id):
            scores = self.data[self.current_mouse_id]
            if scores:
                first_event_frame = min(scores.keys())
                last_event_frame = max((f for f in scores.keys() if f <= self.frame_number), default=first_event_frame)
                
                time_passed = round((self.frame_number - first_event_frame) / self.original_fps, 2)
                time_escaped = round((self.frame_number - last_event_frame) / self.original_fps, 2)
                
                if time_passed > self.session_duration:
                    self.timer_label.config(text=f"Session Time: {time_passed:.2f}s (OVER DURATION)")
                else:
                    self.timer_label.config(text=f"Session Time: {time_passed:.2f}s / {self.session_duration}s")
        else:
            self.timer_label.config(text="Time (s): N/A")

    def update_immobility_stats(self):
        """Your Highly Optimized Immobility Calculation Logic"""
        total_time = 0.0
        current_time = 0.0

        if self.current_mouse_id and self.data.get(self.current_mouse_id):
            data = self.data[self.current_mouse_id]
            frames = sorted(data.keys())
            
            events = [(f, data[f]) for f in frames]
            
            # start from first 0 (Mobility)
            i0 = next((i for i, (_, s) in enumerate(events) if s == 0), None)
            if i0 is None:
                self.total_immobility_label.config(text="0.00 s")
                self.current_immobility_label.config(text="0.00 s")
                return
                
            events = events[i0:]
            start_frame = events[0][0]
            end_limit = start_frame + int(self.session_duration * self.original_fps)

            # build change points
            cp = [events[0]]
            for f, s in events[1:]:
                if s != cp[-1][1]:
                    cp.append((f, s))
                    
            immobile_frames = 0
            current_frames = 0

            for i, (f, s) in enumerate(cp):
                if f >= end_limit:
                    break
                if s == 1:
                    # end of this segment
                    next_f = cp[i+1][0] if i+1 < len(cp) else end_limit
                    seg_start = f
                    seg_end = min(next_f, end_limit)
                    
                    # --- total immobility ---
                    if seg_end > seg_start:
                        immobile_frames += seg_end - seg_start
                        
                    # --- current cumulative immobility (up to frame_number) ---
                    if self.frame_number > seg_start:
                        seg_end_current = min(next_f, self.frame_number, end_limit)
                        if seg_end_current > seg_start:
                            current_frames += seg_end_current - seg_start
                           
            total_time = immobile_frames / self.original_fps
            current_time = current_frames / self.original_fps
                            
        self.total_immobility_label.config(text=f"{total_time:.2f} s")
        self.current_immobility_label.config(text=f"{current_time:.2f} s")

    # ==========================================
    # SAVING & LOADING
    # ==========================================
    def resume_scoring(self):
        self.restore_focus()
        if not self.rois:
            messagebox.showinfo("Info", "No saved ROIs found for this video.", parent=self.tk_window)
            return
    
        def on_select():
            selected_mouse_id = mouse_var.get()
            if selected_mouse_id:
                self.current_mouse_id = selected_mouse_id
                # self.rois[selected_mouse_id] = self.loaded_rois[selected_mouse_id]
                
                # Bug Fix 3: Make sure loaded data is applied to the active scoring dictionary
                if selected_mouse_id in self.loaded_data:
                    self.data[selected_mouse_id] = self.loaded_data.get(selected_mouse_id, {})
                else:
                    self.data[selected_mouse_id] = {}
                    
                self.process_frame()
                self.update_scoring_table()
                self.update_immobility_stats()
            popup.destroy()
            
        popup = tk.Toplevel(self.tk_window)
        popup.title("Resume Scoring")
        popup.geometry("300x150")
        popup.grab_set()
        
        mouse_ids = list(self.rois.keys())
        mouse_var = tk.StringVar(popup, value=mouse_ids[-1])
    
        ttk.Label(popup, text="Select Mouse ID to resume:").pack(pady=10)
        tk.OptionMenu(popup, mouse_var, *mouse_ids).pack(pady=5)
    
        btn_frame = ttk.Frame(popup)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="OK", command=on_select, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=popup.destroy, width=10).pack(side=tk.LEFT, padx=5)

    def load_rois(self):
        self.rois = {}
        file_path = f"{self.video_path.rsplit('.', 1)[0]}_rois.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                self.loaded_rois = {row["Mouse_ID"] : (row["X"], row["Y"], row["Width"], row["Height"]) for _, row in df.iterrows()}
                self.rois = self.loaded_rois
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ROIs: {e}", parent=self.tk_window)

    def load_existing_analysis(self):
        """Bug Fix 3: Ensure data is correctly loaded when video is loaded."""
        self.load_rois()
        if self.rois:
            for mouse_id, _ in self.rois.items():
                file_path = f"{self.video_path.rsplit('.', 1)[0]}_{mouse_id}.xlsx"
                if os.path.exists(file_path):
                    try:
                        df = pd.read_excel(file_path, sheet_name='Immobility')
                        self.loaded_data[mouse_id] = dict(zip(df["Frame"], df["Immobility"]))
                    except Exception as e:
                        print(f"Skipped loading {file_path}: {e}")

    def save_and_next_mouse(self):
        self.restore_focus()
        self.auto_save = True
        self.save_data()
        self.start_roi_drawing()
        self.auto_save = False

    def is_file_open(self, filepath):
        try:
            with open(filepath, "r+"): return False
        except IOError:
            return True

    def save_roi_as_png(self):
        """Saves the current ROI frame as a 600 DPI PNG file."""
        self.restore_focus()
        if self.roi_frame is None or self.video_path is None:
            messagebox.showwarning("Warning", "No video loaded or ROI frame available to save.", parent=self.tk_window)
            return

        try:
            # Convert the frame (which is a NumPy array) to a PIL Image
            pil_img = Image.fromarray(self.roi_frame)

            # Generate a unique filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            mouse_id = self.current_mouse_id if self.current_mouse_id else "FullFrame"
            save_path = f"{self.video_path.rsplit('.', 1)[0]}_{mouse_id}_{self.frame_number}_{timestamp}.png"

            # Save the image with 600 DPI metadata
            pil_img.save(save_path, dpi=(600, 600))
            
            messagebox.showinfo("Success", f"Image saved successfully to:\n{os.path.basename(save_path)}", parent=self.tk_window)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image: {e}", parent=self.tk_window)


    def save_data(self):
        self.restore_focus()
        if not self.data or self.current_mouse_id not in self.data:
            if not self.auto_save: messagebox.showwarning("Warning", "No data to save!", parent=self.tk_window)
            return
    
        base_filepath = f"{self.video_path.rsplit('.', 1)[0]}_{self.current_mouse_id}.xlsx"
    
        if os.path.exists(base_filepath):
            if self.is_file_open(base_filepath):
                messagebox.showerror("Error", f"{base_filepath} is open. Please close it first.", parent=self.tk_window)
                return
            if not self.auto_save:
                response = messagebox.askyesnocancel("File Exists", f"{base_filepath} already exists. Overwrite?", parent=self.tk_window)
                if response is None: return
                if not response:
                    backup_filepath = f"{self.video_path.rsplit('.', 1)[0]}_{self.current_mouse_id}_backup_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
                    shutil.move(base_filepath, backup_filepath)
    
        df = pd.DataFrame(sorted(self.data[self.current_mouse_id].items()), columns=["Frame", "Immobility"])
        df_summary = self.fill_frame_time(df)
        
        if not df_summary.empty:
            df_summary['cum(s)'] = round(df_summary["Immobility"].cumsum() / self.original_fps, 2)
            df_summary['TrialTime'] = df_summary['Time(s)'] - df_summary['Time(s)'].iloc[0]
            
            df_bined = self.bined_immobility(df_summary, 10)
            df_bined_30 = self.bined_immobility(df_summary, 30)
            self.save_to_excel_one(base_filepath, dfs=[df, df_summary, df_bined, df_bined_30], sheet_names=['Immobility', 'Summary', 'dfs_bin_time_df', 'dfs_bin30_time_df'])
        else:
            self.save_to_excel_one(base_filepath, dfs=[df], sheet_names=['Immobility'])

        self.save_rois_to_csv()
        if not self.auto_save:
            messagebox.showinfo("Save", f"Data saved successfully as {base_filepath}!", parent=self.tk_window)

    def save_to_excel_one(self, base_filepath, dfs, sheet_names):
        try:
            wb = Workbook()
            for i, df in enumerate(dfs):
                if not df.empty:
                    data = [df.columns.tolist()] + df.values.tolist()
                    wb.new_sheet(sheet_name=sheet_names[i], data=data)
            wb.save(base_filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save Excel file:\n{e}", parent=self.tk_window)

    def save_rois_to_csv(self):
        filepath = f"{self.video_path.rsplit('.', 1)[0]}_rois.csv"
        existing_mouse_ids = set()
        if os.path.exists(filepath):
            existing_mouse_ids = set(pd.read_csv(filepath)["Mouse_ID"])
            
        new_rois = {mouse_id: roi for mouse_id, roi in self.rois.items() if mouse_id not in existing_mouse_ids}
        if new_rois:
            new_df = pd.DataFrame([(mouse_id, *roi) for mouse_id, roi in new_rois.items()], columns=["Mouse_ID", "X", "Y", "Width", "Height"])
            new_df.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)

    def select_cam_file(self):
        folder_path = os.path.dirname(self.video_path)
        default_filename = None
        match = re.match(r"(.*)_([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}_[0-9]{2}_[0-9]{2})\.(avi|mp4|mov|mkv)", self.video_path, re.IGNORECASE)
        if match:
            base_name, timestamp, _ = match.groups()
            default_filename = f"{base_name}_cam_{timestamp}.csv"
        
        if default_filename and os.path.isfile(os.path.join(folder_path, default_filename)):
            return os.path.join(folder_path, default_filename)
        return filedialog.askopenfilename(parent=self.tk_window, title="Select frame-time CSV file", initialdir=folder_path, filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])

    def fill_frame_time(self, df_data):
        if df_data.empty: return pd.DataFrame()
        start_frame = df_data['Frame'].iloc[0] 
        time_column = 'Time(s)'
        cam_filename = self.select_cam_file()
        
        if cam_filename and os.path.isfile(cam_filename):
            df_raw = pd.read_csv(cam_filename, header=0)
            if 'systemTime' in df_raw.columns: 
                df = df_raw[['systemTime', 'Cam_Frame']].rename(columns={'systemTime': time_column, 'Cam_Frame': 'Frame'})
            else: 
                df = df_raw[['Millis', 'Cam_Frame']].rename(columns={'Millis': time_column, 'Cam_Frame': 'Frame'})
                df[time_column] = df[time_column]/1000
            
            df['Frame'] = df['Frame'] - df['Frame'].iloc[0] + 1 
            start_time = df.loc[df["Frame"] == start_frame, time_column].iloc[0]  
            end_time = start_time + self.session_duration
            df = df[(df[time_column] >= start_time) & (df[time_column] < end_time)].reset_index(drop=True)
        else:
            times = np.round(np.arange(0, self.session_duration, 1 / self.original_fps), 4)
            frames = np.arange(start_frame, start_frame + len(times))
            df = pd.DataFrame({time_column: times, 'Frame': frames})
            
        return self.add_immobility_col(df, df_data)    

    def add_immobility_col(self, df, df_DIO):
        align_column = df_DIO.columns[0]
        DIOs = df_DIO.columns[1:]
        df[DIOs] = df_DIO[DIOs].iloc[-1] 
        for i in range(len(df_DIO) - 1):
            start_frame = df_DIO.at[i, align_column]
            end_frame = df_DIO.at[i + 1, align_column]
            mask = (df[align_column] >= start_frame) & (df[align_column] < end_frame)
            df.loc[mask, DIOs] = df_DIO.loc[i, DIOs].values  
        return df
    
    def bined_immobility(self, df_summary, bin_size): 
        if df_summary.empty: return pd.DataFrame()
        time_column = 'Time(s)'
        df = df_summary.copy()
        df['bin'] = ((df[time_column]-df[time_column].iloc[0]) // bin_size) * bin_size
        immobility_time_per_bin = df.groupby('bin')['Immobility'].sum() * bin_size / df.groupby('bin')['Immobility'].count()
        bin_times_df = pd.DataFrame({'bin': immobility_time_per_bin.index, 'immobility_time': immobility_time_per_bin.values})
        bin_times_df["cum_time"] = bin_times_df["immobility_time"].cumsum()
        return bin_times_df

    # ==========================================
    # CLEANUP & RUN
    # ==========================================
    def toggle_help(self, event=None):
        self.restore_focus()
        if not self.help_visible:
            self.help_window = tk.Toplevel(self.tk_window)
            self.help_window.title("Shortcut list")
            self.help_window.resizable(False, False)
            
            help_text = (
                "Space        - Play/Pause\n"
                "Right Arrow  - Next Frame ⏭\n"
                "Left Arrow   - Previous Frame ⏮\n"
                "R            - Draw New ROI\n"
                "0            - Mark Mobility\n"
                "1            - Mark Immobility\n"
                "Delete       - Remove selected score\n"
                "H            - Toggle Help\n\n"
                "********************************\n"
                "Contact Author:\n"
                "  - Zengyou Ye\n"
                "  - addy9908@gmail.com"
            )
            tk.Label(self.help_window, text=help_text, justify=tk.LEFT, anchor="w").pack(padx=20, pady=20)
            self.help_visible = True
            self.help_window.protocol("WM_DELETE_WINDOW", self.toggle_help)
        else:
            if hasattr(self, 'help_window') and self.help_window.winfo_exists():
                self.help_window.destroy()
            self.help_visible = False

    def on_close(self):
        self.playing = False
        if messagebox.askokcancel("Quit", "Do you want to quit? Unsaved data will be lost.", parent=self.tk_window):
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.tk_window.quit()
            self.tk_window.destroy()

    def run(self):
        self.tk_window.mainloop()

if __name__ == "__main__":
    app = VideoScoring()
    app.run()
