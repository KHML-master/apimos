import tkinter.filedialog
import tkinter as tk

import configparser
from PIL import Image, ImageTk, ImageDraw
import os
import sys

import json

from pathlib import Path
import numpy as np
import pickle

# Import calibration module
path = os.path.abspath(Path('../../src/position_estimation'))
sys.path.insert(1, path)
import calibration


class CalibrationMatrix:
    def __init__(self, root, config):
        self.root = root
        self.dots = []
        self.txts = []
        self.dot_width = 5
        self.config = config

    def create(self, img_name):
        self.img_name = img_name

        # Image
        img = Image.open(self.img_name)
        self.im_width, self.im_height = img.size
        self.img = ImageTk.PhotoImage(img)
        self.root.one = self.img  # Prevent Garbage Collection

        # Canvas
        self.canvas = tk.Canvas(width=self.im_width, height=self.im_height)
        self.canvas.pack(side='top', fill='both', expand=True)
        self.canvas.create_image((self.im_width/2 , self.im_height/2), image=self.img, anchor='center')

        self.canvas.bind('<Button-1>', self.__set_point)  # Left Mouse Press
        self.root.bind('<Return>', self.__finish_calibration)
        self.root.bind_all("<Control-z>", self.__redo)

    def __set_point(self, event):
        print(f'[INFO] Setting Point {len(self.dots)+1}')
        
        # Pos of Mouse Press
        x1, y1 = (event.x - self.dot_width), (event.y - self.dot_width)
        x2, y2 = (event.x + self.dot_width), (event.y + self.dot_width)
        xc = x1 + self.dot_width
        yc = y1 + self.dot_width

        # Dot Creation
        dot = self.canvas.create_oval(x1, y1, x2, y2, fill='red')  # Draw Dot
        txt = self.canvas.create_text(xc, yc, text=str(len(self.dots)+1), anchor=tk.CENTER, fill='white')
        self.dots.append(dot)
        self.txts.append(txt)

    def __finish_calibration(self, event):
        if len(self.dots) != 8:
            print('[WARNING] You need to have 8 points total.')
            return

        print('[INFO] Finished Calibation')
        data = {}
        for idx, d in enumerate(self.dots):
            x1, y1, _, _ = self.canvas.coords(d)
            xc = x1 + self.dot_width
            yc = y1 + self.dot_width
            data[idx] = [xc, yc]

        cal_mat = calibration.CalibrationJSON(config=config)
        cal_mat.calibrate_camera(data)
        quit()
    
    def __redo(self, event):
        if len(self.dots) > 0:
            print('[INFO] Delete previous point')
            self.canvas.delete(self.dots[-1])
            self.canvas.delete(self.txts[-1])
            self.dots.pop()
            self.txts.pop()

if __name__ == '__main__':
    print('___Create Calibation Matrix___\nCreate point: left mouse button\nRedo point: CTRL-Z\nFinish calibration: ENTER\n______________________________')
    config = configparser.ConfigParser()
    config.read('./pig_pen.config')


    save_path = f'../../src/position_estimation/data_calibration/cal_matrix/{config["camera"]["name"]}.npy'
    if os.path.isfile(save_path):
        print(f'[WARNING] The file: ../../src/position_estimation/data_calibration/cal_matrix/{config["camera"]["name"]}.npy already exists')

    root = tk.Tk()

    try:
        img_name = list(Path('').glob('*.jpg'))[0]
    except Exception as err:
        print('[ERROR] You need an JPG image located in calibration_matrix directory.') 
        quit()

    cal_mat = CalibrationMatrix(root=root, config=config)
    cal_mat.create(img_name=img_name)

    root.mainloop()
