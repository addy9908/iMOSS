# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:59:59 2025

Rules:
1. an immobility must last more than 1.8 secs (isi_threshold)
2. setting:
    self.session_duration = 360
    self.threshold = 0.792 from ML

@author: yez4
"""

#%% import libraries
import os, sys,time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from pyexcelerate import Workbook
import tkinter as tk
from tkinter import messagebox
from scipy import signal
from zy_importer_lite_V4 import FileSelectorWindow
from zy_preset_mpl_v2 import preset_mpl

#%% Class detect_immobility
class detect_immobility:
    def __init__(self):
        self.Author = 'Zengyou Ye'
        self.version = 'zy_tailsuspension_V13' 
        preset_mpl()
        
        self.session_duration = 360
        self.BL_points = 800 # 20s,change if too small #(on UI?)
        self.rolling_size = 250  # Rolling window size for std
        self.time_column = 'Time(s)'
        self.save_files = 1 #(on UI)
        self.save_PDF = 1 #(on UI)
        self.isi_threshold = 1.8
        self.threshold = 0.792
        self.lowcut = 1
        self.highcut = 20 #bandfilter
        self.bin_size = 10 # calculate mobile time per 10s
        self.auto_detect = 1
        self.Provisional_start_time = None
        self.start_time = None # system time
        self.abs_start_time = None # time from start
        self.preview_startTime = 1
        self.debug = 0
        self.signal_color = 'r'
        self.BL_color = 'lime'
        self.highlight_color = 'magenta'
        self.mobile_color = 'r'#'pink'#'cyan'
        self.immobile_color = 'b'#'cyan'#'lime'#'plum'#(110/255, 185/255, 210/255) #'blue'        
              
    def initial_parameters(self): #outputs
        self.filename = None
        self.basename  = None
        self.file_path = None
        self.output_path = None
        self.formatted_time = None
        self.time_step = None
        self.baseline = None
        self.baseline_mask = None

        self.end_time = None
        self.BW_filter = None
        self.period_threshold = None
        self.latency = 0
        
        self.total_mobile_time = None
        self.total_immobile_time = None
        self.last4_immobile_time = None
        
        self.figs={}
        self.dfs={}
     
    def filename_refill(self,filename):
        if not os.path.isabs(filename):
            # If the filename is not an absolute path, get the current working directory and join the filename to it
            filename = os.path.join(os.getcwd(), filename)
        
        return filename

    def filterSignal(self,data,filter_window=100):
        if filter_window in (0,1): # I will use 100 for 80Hz sampling rate for over-smooth
            return data
 
        b = np.divide(np.ones((filter_window,)), filter_window)
        a = 1
        filtered_signal = signal.filtfilt(b, a, data) #used to apply a digital filter to a sequence of data using the forward-backward filtering method
        return filtered_signal
    
    def bandpass_filter(self, data, lowcut=1, highcut=6, order=4):
        fs = int(1/self.time_step)    # Sampling frequency (Hz) — adjust if different

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, data) #used to apply a digital filter to a sequence of data using the forward-backward filtering method
        return filtered_signal
    
    def highpass_filter(self, data, cutoff=20.0, order=4):
        # Nyquist frequency is half the sampling rate
        fs = int(1/self.time_step) 
        nyq = 0.5 * fs
        # Normalized cutoff frequency (cutoff / Nyquist)
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients for a highpass filter
        b, a = signal.butter(order, normal_cutoff, btype='high')
        # Apply the filter to the data (zero-phase filter)
        filtered_signal = signal.filtfilt(b, a, data)
        return filtered_signal
        
    def get_startTime_from_preview(self,df,start_time):
        # Initialize start_time with the first time value
        time_column = self.time_column
        start_time = start_time if start_time else df[time_column].iloc[0]
        
        # Function to update vline dynamically as user types
        def update_vline(event=None):
            try:
                ax.lines[-1].remove()  # Remove previous vline
                ax.axvline(trim_start_var.get(), color=self.BL_color, linestyle='--', label="Start Time")
                canvas.draw()
                selected_option.set(options[0])
            except ValueError:
                pass  # Ignore invalid input
        
        # Function to finalize start_time on submit
        def submit():
            try:
                self.start_time = trim_start_var.get()  # Get input value
                self.threshold = threshold_factor_var.get()
                detect_option = selected_option.get()
                mapping = {'keep manual': 0, 'keep auto': 1,'apply to all': 2}              
                self.auto_detect=mapping.get(detect_option, None)
                self.preview_startTime = 0 if self.auto_detect else 1
                self.save_files = save_files_var.get()
                self.debug = debug_var.get()
                self.save_PDF = save_pdf_var.get()
                root.destroy()
                            
            except ValueError:
                print("Invalid input! Please enter a numerical value.")
                
        def on_quit_button_click():  # need to clear back to default
            print('User quit the selector')
            root.destroy()
            sys.exit()  # stop the Python script
                    
        root = tk.Tk()
        root.attributes("-topmost", True)
        root.title("Choose your start time")
        
        # Create Matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df[time_column], df["Signal"], label="Raw Data", color=self.signal_color)
        ax.set_xlabel(self.time_column)
        ax.set_ylabel("Signal")
        ax.set_xlim(df[time_column].min(),df[time_column].max())
        ax.set_title(self.basename)
        ax.grid()
        # Initial vline at first time point
        ax.axvline(start_time, color=self.BL_color, linestyle='--', label="Start Time")
        
        # Embed plot in Tkinter
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        canvas = FigureCanvasTkAgg(fig, master=top_frame)
        # Create the NavigationToolbar2Tk for zoom and save
        toolbar = NavigationToolbar2Tk(canvas, top_frame)
        toolbar.update()
        # Pack the toolbar on top of the canvas
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # Pack the canvas below the toolbar
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)#, expand=True) 
        
        # Input field
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(side=tk.BOTTOM, padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        trim_start_label = tk.Label(bottom_frame, text="Start_Time (s):",fg="red")
        trim_start_label.grid(row=0, column=0, sticky='nsew')
               
        trim_start_var = tk.DoubleVar() 
        trim_start_var.set(start_time)        
        trim_start_entry = tk.Entry(
            bottom_frame, textvariable=trim_start_var, width=6)
        trim_start_entry.grid(row=0, column=1, sticky='nsew')
        trim_start_entry.bind("<KeyRelease>", update_vline)
        

        threshold_factor_label = tk.Label(bottom_frame, text="fixed threshold:",fg="red")
        threshold_factor_label.grid(row=1, column=0, sticky='nsew')
               
        threshold_factor_var = tk.DoubleVar() 
        threshold_factor_var.set(self.threshold)        
        threshold_factor_entry = tk.Entry(
            bottom_frame, textvariable=threshold_factor_var, width=6)
        threshold_factor_entry.grid(row=1, column=1, sticky='nsew')
        
        # options
        options = ['keep manual','keep auto','apply to all']
        select = options[self.auto_detect]
        selected_option = tk.StringVar()
        selected_option.set(select)  
        option_label = tk.Label(bottom_frame, text="detection option:")
        option_label.grid(row=0, column=2, sticky='nsew')
        option_menu = tk.OptionMenu(bottom_frame, selected_option, *options)
        option_menu.grid(row=0, column=3, sticky='nsew')
        
        # save option
        # Checkbox for save_files
        save_files_var = tk.BooleanVar()
        save_files_var.set(self.save_files)  # Default is not including subfolders
        save_files_checkbox = tk.Checkbutton(
            bottom_frame, text="save files", variable=save_files_var,fg='red')
        save_files_checkbox.grid(row=1, column=2, sticky='nsew')  
        
        # Checkbox for save_PDF
        save_pdf_var = tk.BooleanVar()
        save_pdf_var.set(self.save_PDF)  # Default is not including subfolders
        save_pdf_checkbox = tk.Checkbutton(
            bottom_frame, text="save fig as PDF", variable=save_pdf_var,fg='red')
        save_pdf_checkbox.grid(row=1, column=3, sticky='nsew') 
        
        # Checkbox for debug
        debug_var = tk.BooleanVar()
        debug_var.set(self.debug)  # Default is not including subfolders
        debug_checkbox = tk.Checkbutton(
            bottom_frame, text="debug?", variable=debug_var,fg='red')
        debug_checkbox.grid(row=2, column=2, sticky='nsew') 
        
        # Buttom
        submit_button = tk.Button(bottom_frame, text="Submit", command=submit)
        submit_button.grid(row=2, column=3, sticky='se')
        
        plt.close(fig) #'all'
        
        # Run Tkinter loop
        bottom_frame.columnconfigure(0, weight=1)  # Allow expansion of column 0  
        bottom_frame.columnconfigure(1, weight=1)
        bottom_frame.columnconfigure(2, weight=1)
        bottom_frame.columnconfigure(3, weight=1)
        root.protocol("WM_DELETE_WINDOW", on_quit_button_click)
        root.mainloop()

    def detailed_extract_signal(self,df):

        # --- 1. manipulate data  ---    
        time_column = self.time_column
        df['filteredSignal'] = self.bandpass_filter(df['Signal'],lowcut = self.lowcut, highcut = self.highcut, order =4)
        
        # 2. Calculate the rolling standard deviation
        df['RollingStd'] = df['Signal'].rolling(window=self.rolling_size,  min_periods=1, center=True).std()
        # 3. get dynamic BW_filter by filter_window at 3000
        flat_signal = pd.Series(self.filterSignal(df['Signal'],filter_window = 100))
        df['flat_signal'] = flat_signal
        
        #--- 2. pretrim with period_threshold for pre and post-start periods---
        period_threshold = flat_signal.mean()*0.9 + flat_signal.min()*0.1 # period threshold
        self.period_threshold = period_threshold
        print(f'period_threshold={period_threshold}')
        
        # find baseline from the middle third smallest RollingStd
        def find_baseline(df,nPoints=800): #10s with 80fps
            df_middle = df.iloc[len(df) // 3:-len(df) // 3]  # Slice the DataFrame to get middle 1/3   
            # baseline from 160 points with smallest RollingStd
            baseline_mask = df_middle.nsmallest(nPoints, 'RollingStd').index
            
            baseline = df_middle.loc[baseline_mask, 'Signal'].mean()
            self.baseline_mask = baseline_mask
            return baseline
                    
        # --- 3.1. cut off long waiting baseline below BW_filter and find baseline
        start_time = df[time_column].iloc[0]
        if self.auto_detect ==1: #keep auto
            try: 
                continue_signal_check = int(270/self.time_step) # keep higher for 270s
                flat_mask = (flat_signal>period_threshold)[::-1].rolling(window=continue_signal_check).sum()[::-1] == continue_signal_check #rolling from right side
                period_separating_index = flat_mask.idxmax() if flat_mask.any() else 0                
                Provisional_start_time = df[time_column].iloc[period_separating_index]
                self.Provisional_start_time = Provisional_start_time

            except:
                print('Error: period_threshold is too large')
                period_separating_index=df.index[0]
            
            df_trim = df.loc[period_separating_index:]        # do not reindex since we need to plot baseline_mask   
            
            baseline = find_baseline(df_trim,nPoints=self.BL_points)
            
            try: #if no baseline
                start_time = df_trim.loc[df_trim['Signal'] > baseline, time_column].iloc[0]
                print(f'Auto start time={start_time}')
            except:                
                print(f'wrong baseline = {baseline}')
                pass
            
            # assign the self.start_time
            if self.preview_startTime:
                self.get_startTime_from_preview(df,start_time) # get self.start_time 
            else:
                self.start_time = start_time

        elif self.auto_detect ==2: #apply to all
            # do not overwrite the self.start_time
            print(f'reuse start_time to all at: {self.start_time}')
            
        # 4.2. if fail or default,ask user to confirm the start_time, 
        # decide to use it to all files or continue auto_detect
        elif (not self.auto_detect):  # 0. keep manual
            # get updated self.start_time and self.auto_detect
            self.get_startTime_from_preview(df,start_time) # get self.start_time  
        
        print(f"Final start_time = {self.start_time}")
        
        #5. final extraction and recalculate baseline
        df_extracted = df[(df[time_column]>= self.start_time) & (df[time_column]<= self.start_time+self.session_duration)]
        baseline = find_baseline(df_extracted,nPoints=self.BL_points) # calibrate baseline after extraction
        
        df_extracted.reset_index(drop=True, inplace=True)                  
        self.baseline = baseline
        self.abs_start_time = self.start_time - df[time_column].iloc[0]
        print (f'calculated baseline is {baseline}')
        
        # subtract baseline
        df_extracted.loc[:,'Signal'] = df_extracted.loc[:,'Signal']  - baseline
        # df_extracted.loc[:,'filteredSignal'] = df_extracted.loc[:,'filteredSignal'] #- baseline (no need for band-pass)
        
        #%% add plot raw here
        if self.auto_detect ==1 and self.debug: # plot in more details
            print('Debug: plot the extraction process')
            fig = plt.figure(figsize=(6, 8))
            gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 1], 
                                   wspace=0.02, hspace = 0.3, left = 0.1, right = 0.95, 
                                   bottom = 0.1, top = 0.90)
            # raw
            ax1 = fig.add_subplot(gs[0])
            line1, = ax1.plot(df[time_column].iloc[period_separating_index:], df['Signal'].iloc[period_separating_index:], 
                         color='r',
                         label='4. Separated post-start period')
            
            ax1.plot(df[time_column].iloc[:period_separating_index], df['Signal'].iloc[:period_separating_index], 
                          color='gray', linestyle='--', 
                          label = 'separated pre-start period')
            
            line5 = ax1.axhline(y=self.baseline, color='deepskyblue', linestyle='--', label=f'7. Calculate baseline based on middle {self.BL_points} smallest std')
        
            ax1.plot(df[time_column][self.baseline_mask], df['Signal'][self.baseline_mask], 
                              marker='o', color=self.highlight_color, markersize=1, linestyle='None')
            
            line6 = ax1.axvline(x=self.start_time, color='purple', linestyle='--', 
                                label = f"8. Refine start time with calculated baseline: $T_{{start}}$ = {round(self.start_time, 2)}")
            
            ax1.set_xlim(df[time_column].iloc[0],df[time_column].iloc[-1])
            ax1.set_xlabel("")
            ax1.set_ylabel("Raw Signal",fontweight='bold')
            
            # oversmoothed
            ax10 = fig.add_subplot(gs[1], sharex=ax1)
            ax10.plot(df[time_column], flat_signal, label='1. Over-Smoothed Signal (filtfilt with windows size 100)',color='purple')
            ax10.axhline(y=period_threshold, color='k', linestyle='--', label = f'2. Estimate adaptive period threshold={round(period_threshold,2)}')
            ax10.axvline(Provisional_start_time,color='b', linestyle='--', label = f'3. Determine provisional start time = {round(Provisional_start_time,2)}')
            ax10.set_xlabel("")
            ax10.set_ylabel("Smoothed Signal",fontweight='bold')
            ax10.set_ylim(ax1.get_ylim())
            
            #rolling std
            ax2 = fig.add_subplot(gs[2], sharex=ax1)
            ax2.plot(df[time_column], df['RollingStd'], label=f'5. Calculate rollingStd of signal (window_size = {self.rolling_size})')           
            
            #hilight the parts used for calculating baseline
            ax2.plot(df[time_column][self.baseline_mask], df['RollingStd'][self.baseline_mask], 
                               marker='o', color=self.highlight_color, markersize=1, linestyle='None', 
                               label=f'6. Pick {self.BL_points} smallest-Std points within the middle third of the post-start period')

    
            
            ax2.set_xlabel("")
            ax2.set_ylabel("RollingStd",fontweight='bold')
            
            # extracted signal                    
            ax3 = fig.add_subplot(gs[3],sharex=ax1)
            ax3.plot(df_extracted[time_column], df_extracted['filteredSignal'], 
                               label=f'9.Extracted 6-min Signal followed by baseline subtraction and band filtered ({self.lowcut}-{self.highcut} Hz)', color='red')
            ax3.axvline(x=df_extracted[time_column].iloc[0], color='purple', linestyle='--')
            ax3.axvline(x=df_extracted[time_column].iloc[-1], color='purple', linestyle='--')
            
            ax3.set_xlabel("Time (s)",fontweight='bold')
            ax3.set_ylabel("Extracted Signal",fontweight='bold')
            ax1.tick_params(labelbottom=False)
            ax10.tick_params(labelbottom=False)
            ax2.tick_params(labelbottom=False)
            
            ax1.legend(handles=[line1,line5,line6],loc='upper right')
            ax10.legend(loc='upper right')
            ax2.legend(loc='upper right')
            ax3.legend(loc='upper right')
            
            fig.align_labels()
            fig.suptitle(self.basename, fontsize = 10)
            plt.show()
            self.figs['debug_view'] = fig                       
            
        else: #just plot the signal/filtered signal around start_time 
            print('plot signal/filtered signal around start_time')
            fig, ax = plt.subplots(figsize=(6, 3))
            mask = df[time_column]<=self.start_time +10
            ax.plot(df[time_column].loc[mask], df['Signal'].loc[mask], 
                         color='r',
                         label='Raw')
            
            ax.plot(df[time_column].loc[mask], df['filteredSignal'].loc[mask], 
                          color='blue',  
                          label = 'Filtered')
            
            ax.axvline(x=self.start_time, color='purple', linestyle='--', label = f'$T_{{start}}$={round(self.start_time,2)}')  # vertical line at start time
            ax.axhline(y=self.baseline, color='deepskyblue', linestyle='--', label=f'Baseline={round(self.baseline,2)}')
            
            ax.set_xlabel("Time (s)",fontweight='bold')
            ax.set_ylabel("Signal",fontweight='bold')
            ax.legend(loc='upper right',fontsize = 12)            
            ax.set_title(self.basename, fontsize = 10)
            plt.tight_layout()
            plt.show()
            self.figs['brief_view'] = fig     
        
        #%% reset time from 0 and subtract baseline
        df_extracted.loc[:,time_column] =  df_extracted.loc[:,time_column] - df_extracted.loc[0,time_column] 
        
        return df_extracted
                    
    def save_all_to_excel_20250203(self, skip=[]):
        print("Save all properties of self instance to excel:")
        excel_filename = os.path.join(self.output_path,self.basename.split('.')[0] + '_summary_' + self.formatted_time +'.xlsx')
        filename = os.path.basename(excel_filename)
        
        def flatten_defaultdict(d, parent_key='', sep='_'): # also work with dict
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_defaultdict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        ''' check all properties'''
        filtered_dict = {k: v for k, v in self.__dict__.items() if k not in skip}
        flatten_dict = flatten_defaultdict(filtered_dict)
        
        # Create a new workbook and define worksheet name
        def excel_writer():
            wb = Workbook()
            log = [["Properties", "Value"]]  # Header row
            
            for k,value in flatten_dict.items():                        
                if isinstance(value,pd.DataFrame) and not value.empty:
                    data = [value.columns.tolist(),] + value.values.tolist() # with header
                    wb.new_sheet(sheet_name =k, data=data)
                    
                elif isinstance(value, (str, bool, int, float)):
                    log += [[k,value]]
                    
                elif isinstance(value, mpl.figure.Figure):
                    print('Figures will be saved seperately')
            
            wb.new_sheet(sheet_name = 'Log', data=log)
            wb.save(excel_filename)
        
            
        while True:
            try:
                excel_writer()
                print('>>>>>>excel saved: ',filename)
                break # Exit the loop if writing is successful
            except PermissionError:
                # If a PermissionError occurs, prompt the user to close the Excel file and continue
                # easygui.msgbox("Please close the Excel file'{}', then click OK to continue.".format(filename), title="File in Use")
                root = tk.Tk()
                root.withdraw()  # Hide the root window
                
                # Create a hidden root window and bring it to the top
                root.attributes("-topmost", True)
                root.update()  # Force update to apply the topmost attribute
                
                # Now show the messagebox
                messagebox.showinfo("File in Use", f"Please close the Excel file '{filename}', then click OK to continue.", parent=root)
                root.destroy()  # Destroy the root window after the messagebox is closed          
    
    def plot_detection(self,df,used_signal_col):
        # used_signal_col = 'filteredSignal'
        time_step = self.time_step
        time_column = self.time_column
        used_immobile_col = 'Immobile_filter' if used_signal_col =='filteredSignal' else 'Immobile'
        # used_cum_col = 'Cumulative_Immobility_filter' if used_signal_col =='filteredSignal' else 'Cumulative_Immobility'
        threshold_line = self.threshold
        
        # sum works better and more efficiency here
        total_immobile_time = round(df[used_immobile_col].sum() * time_step,2)
        # total_mobile_time = round(df[time_column].iloc[-1]-total_immobile_time,2)
        total_mobile_time = round(self.session_duration - total_immobile_time,2)
        
        self.total_mobile_time = total_mobile_time
        self.total_immobile_time = total_immobile_time
        self.last4_immobile_time = round(df[used_immobile_col].loc[df[time_column] >=120].value_counts().get(1, 0) * time_step,2)        
        
        fig, ax1 = plt.subplots(figsize=(6, 3))

        ''' 1. line plots'''
        ax1.plot(df[time_column], df[used_signal_col], label=used_signal_col, color=self.signal_color)
        
        line4= ax1.axhline(y=threshold_line, color=self.BL_color, linestyle='--', label = f'Threshold={round(threshold_line,3)}')
        ax1.axhline(y=-threshold_line, color=self.BL_color, linestyle='--')
        
        # Mask: Loop over segments, but only where changes occur
        change = df[used_immobile_col].diff().fillna(0)
        change_points = change[change != 0].index.to_list()
        self.latency = df.loc[change_points[0],self.time_column] if change_points else self.session_duration # will be 360 if no immobility
        print(f'latency is {self.latency}')
        
        change_points.insert(0, 0)  # Start of first segment
        change_points.append(len(df))  # End of last segment
        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i + 1]
            color = self.mobile_color if df[used_immobile_col].iloc[start_idx] == 0 else self.immobile_color
            ax1.axvspan(df[time_column].iloc[start_idx], df[time_column].iloc[end_idx-1], alpha=0.2, color=color) #, alpha=0.3
            # if df[used_immobile_col].iloc[start_idx] == 1:
            #     ax1.axvspan(df[time_column].iloc[start_idx], df[time_column].iloc[end_idx-1], alpha=0.7, color=self.immobile_color)
            
        # Create dummy artists (patches) for the legend
        patch_red = mpatches.Patch(color=self.mobile_color, label=f'Mobility: {total_mobile_time} s',alpha=0.2)
        patch_blue = mpatches.Patch(color=self.immobile_color, label=f'Immobility: {total_immobile_time} s',alpha=0.2)
        
        ax1.set_xlim(0,self.session_duration)
        ax1.set_xticks(range(0, self.session_duration+1, 60))
        # minor_locator = mpl.ticker.AutoMinorLocator(2) # Places 1 minor tick midway between major ticks
        # ax1.xaxis.set_minor_locator(minor_locator)
        ax1.set_xlabel("Time (s)",fontweight='bold')
        ax1.set_ylabel(used_signal_col,fontweight='bold')
        ax1.set_title(self.basename, fontsize = 10)
        ax1.legend(handles=[line4, patch_blue],loc='upper right',fontsize = 12) #line1,line2,line3,
        

        plt.tight_layout()
        plt.show()        
        self.figs['Immobility_detection'] = fig
    
    def run(self,file):
        plt.close('all')
        self.initial_parameters()
        self.filename = self.filename_refill(file) #also get self.filename here
        self.basename = os.path.basename(self.filename)
        print(f'Run on file: {self.basename}')
        self.file_path = os.path.dirname(self.filename)
        # self.root_path = os.path.dirname(self.file_path)
        time_column = self.time_column
        
        # setup output filenames
        date_str = datetime.now().strftime("%m%d%Y")
        self.output_path = os.path.join(self.file_path, f"Output_final_{date_str}")
        os.makedirs(self.output_path, exist_ok=True)  # more advance than check the isdir
        self.formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        #%% 1. load file
        df_loadcell = pd.read_csv(self.filename, header=0)
        
        if 'systemTime' in df_loadcell.columns: # neurophotometrics time at sec
            df_raw = df_loadcell[['systemTime', 'Loadcell']].rename(columns={'systemTime': time_column, 'Loadcell': 'Signal'})
        else: # computer time at ms
            df_raw = df_loadcell[['Millis', 'Loadcell']].rename(columns={'Millis': time_column, 'Loadcell': 'Signal'})
            df_raw[time_column] = df_raw[time_column]/1000

        # Convert all columns to numeric, invalid parsing will be set as NaN
        df_raw = df_raw.apply(pd.to_numeric, errors='coerce')

        # df_raw[time_column] = df_raw[time_column] / 1000  # Convert milliseconds to seconds
        df_raw['Signal'] = df_raw['Signal'] / 1000 #convert from mg to g
        time_step = df_raw[time_column].diff().mean().item()
        self.time_step = time_step
        
        #%% 2. extract 6min signal
        '''
        1. use auto first, then preview the result: plot with start time
        2. agree, click auto. Otherwise set the start time (edit) and click manual
        '''
        df = self.detailed_extract_signal(df_raw) # insert plot raw here
        df = df.copy() # Make sure we're working with a standalone copy
        
        #%%3. Spike detection on original signal

        threshold = self.threshold        
        #3.1  Create binary spike train based on the computed threshold
        spike_train = (df['Signal'].abs() > threshold).astype(int)   

        
        #3.2 assign immobility as 1 based on signal
        # Step 1: Get spike times where spikes occur
        spike_times = df.loc[spike_train == 1, time_column].to_numpy()
        
        # Step 2: Compute ISI (Inter-Spike Intervals)
        isi = np.diff(spike_times)
        
        # Step 3: Initialize 'immobility' column to 1
        df['Immobile'] = 1
        
        # Step 4: Assign mobility = 1 between consecutive spikes if ISI < threshold
        isi_threshold = self.isi_threshold  # Define burst interval threshold
        
        for i in range(len(isi)):
            if isi[i] < isi_threshold:
                df.loc[(df[time_column] >= spike_times[i]) & 
                       (df[time_column] <= spike_times[i + 1]), 'Immobile'] = 0
        
        # Step 5: Ensure the first spike has immobility = 0
        df.loc[df[time_column] < spike_times[0], 'Immobile'] = 0
        
        # do not forget the last point 20250411
        if (self.session_duration - spike_times[-1])< isi_threshold:
            df.loc[df[time_column]>spike_times[-1],'Immobile'] = 0 
             
        df['Cumulative_Immobility'] = df['Immobile'].cumsum()*time_step
        
        #%% 4. Spike detection on original signal
        spike_train2 = (df['filteredSignal'].abs() > self.threshold).astype(int) 
        spike_times2 = df.loc[spike_train2 == 1, time_column].to_numpy()
        
        # Step 2: Compute ISI (Inter-Spike Intervals)
        isi2 = np.diff(spike_times2)
        
        # Step 3: Initialize 'immobility' column to 1
        df['Immobile_filter'] = 1
        
        # Step 4: Assign mobility = 1 between consecutive spikes if ISI < threshold

        
        for i in range(len(isi2)):
            if isi2[i] < isi_threshold:
                df.loc[(df[time_column] >= spike_times2[i]) & 
                       (df[time_column] <= spike_times2[i + 1]), 'Immobile_filter'] = 0
        
        # Step 5: Ensure the first spike has immobility = 0
        df.loc[df[time_column] < spike_times2[0], 'Immobile_filter'] = 0
        
        # do not forget the last point 20250411
        if (self.session_duration - spike_times2[-1])< isi_threshold:
            df.loc[df[time_column]>spike_times2[-1],'Immobile_filter'] = 0 
        
        df['Cumulative_Immobility_filter'] = df['Immobile_filter'].cumsum()*time_step
        
        ''' 5. calculate bin-timed mobility and immobility'''
        # Step 1: Create bins based on time_column and bin_size
        bin_size = self.bin_size #sec
        df['bin'] = (df[time_column] // bin_size) * bin_size
        
        # Step 2: Calculate total immobility time in each bin using sum
        # using sum/count could be more accuracy then sum * time_step 
        # immobility_time_per_bin = df.groupby('bin')['Immobile'].sum() * bin_size / df.groupby('bin')['Immobile'].count()
        immobility_time_per_bin = df.groupby('bin')['Immobile'].sum() * self.time_step
        immobilityF_time_per_bin = df.groupby('bin')['Immobile_filter'].sum() * self.time_step
        
        # Step 4: Combine the results into a DataFrame
        bin_times_df = pd.DataFrame({
            'bin': immobility_time_per_bin.index,  # The start of each bin
            'immobility_time': immobility_time_per_bin.values,
            'immobility_filter_time': immobilityF_time_per_bin.values
        })
        
        ''' calculate default bin-timed mobility and immobility at 30s'''
        # Step 1: Create bins based on time_column and bin_size
        default_bin_size = 30
        df['default_bin'] = (df[time_column] // default_bin_size) * default_bin_size
        
        # Step 2: Calculate total mobility time in each bin using sum
        immobility_time_per_bin30 = df.groupby('default_bin')['Immobile'].sum() * self.time_step
        immobilityF_time_per_bin30 = df.groupby('default_bin')['Immobile_filter'].sum() * self.time_step
        
        # Step 4: Combine the results into a DataFrame
        bin30_times_df = pd.DataFrame({
            'bin': immobility_time_per_bin30.index,  # The start of each bin
            'immobility_time': immobility_time_per_bin30.values,
            'immobility_filter_time': immobilityF_time_per_bin30.values
        })
        bin30_times_df["cum_time"] = bin30_times_df["immobility_time"].cumsum()
        bin30_times_df["cum_filter_time"] = bin30_times_df["immobility_filter_time"].cumsum()
        
        #%% 5. Plot data
        self.plot_detection(df, 'filteredSignal')
         
        ''' 3. save files''' 
        df['raw_time'] = df[time_column] + self.start_time
        self.dfs['Raw'] = df_raw
        self.dfs['Filtered_data'] = df
        self.dfs['Bin_time_df'] = bin_times_df
        self.dfs['Bin30_time_df'] = bin30_times_df

        
        if self.save_files:                   
            self.save_all_to_excel_20250203(skip=['figs'])
            if self.save_PDF:
                pdf_filename = os.path.join(self.output_path, self.basename.split('.')[0] +'_'+ self.formatted_time +'.pdf')
                with PdfPages(pdf_filename) as pdf:
                    for event in self.figs.keys():
                        pdf.savefig(self.figs[event])
            
            else:
                for event in self.figs.keys():
                    png_filename = os.path.join(self.output_path,self.basename.split('.')[0] + f'_{event}_' + self.formatted_time +'.png')
                    self.figs[event].savefig(png_filename, dpi=600)

#%% main
if __name__ == "__main__":
    plt.close('all')   
    
    #%% 1. select files
    file_selector = FileSelectorWindow()
    # preset
    file_selector.default_path = os.getcwd() # change to your directory if needed
    file_selector.extension_choice='csv'
    file_selector.include_subfolders=True

    # open the popup window for you to choose
    file_selector.select()
    selected_files = file_selector.selected_files

    #%% 2. preset the detect_immobility
    TST = detect_immobility()
    TST.save_PDF = 1
    TST.auto_detect = 1
    TST.save_files = 0 if TST.debug else 1
    
    #%% 3. run detect_immobility
    start_time = time.time()
    for file in list(selected_files):
        TST.run(file)   
        
    end_time = time.time()
    elapsed_time = end_time-start_time
    print(f'Elapsed time for {len(selected_files)} files: {elapsed_time} seconds')






