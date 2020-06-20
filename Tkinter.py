# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:43:24 2020

@author: user
"""

import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from PIL import ImageTk, Image
from tkinter import filedialog
from test_model import Output
import numpy as np
import cv2
import re

class Mouse:
    def __init__(self,canvas):
        
        self.x = self.y = 0
        self.canvas = canvas

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.rect = None

        self.start_x = None
        self.start_y = None
        
        self.mask_img = np.zeros((256,256), dtype="uint8")

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create rectangle if not yet exist
        #if not self.rect:
        self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, fill='white', outline="")

    def on_move_press(self, event):
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)
        
        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)    

    def on_button_release(self, event):
        self.rectRecord(self.start_x,self.start_y,self.canvas.canvasx(event.x),self.canvas.canvasy(event.y))
        pass
    
    def rectRecord(self,x1,y1,x2,y2):
        print(x1,y1,x2,y2)
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        for i in range(y1,y2+1):
            for j in range(x1,x2+1):
                self.mask_img[i,j] = 255

def readFile():
   #print("readFile")
   global img
   global mouse
   global pic
   mouse = Mouse(canvas)
   basewidth = 700
   baseheight = 500
   imgsize = 0,0
   try:
       filename = filedialog.askopenfilename(title='open')
       img = Image.open(filename)
       '''
       if img.size[0]>img.size[1]:
           wpercent = (basewidth/float(img.size[0]))
           hsize = int((float(img.size[1])*float(wpercent)))
           imgsize = basewidth, hsize
           img = img.resize((basewidth, hsize), Image.ANTIALIAS)

       else:
           wpercent = (baseheight/float(img.size[1]))
           wsize = int((float(img.size[0])*float(wpercent)))
           imgsize = wsize,baseheight
           img = img.resize((wsize, baseheight), Image.ANTIALIAS)
       '''
       print(img.size)
       canvas.config(width=img.size[0],height=img.size[1])
       img = ImageTk.PhotoImage(img)
       canvas.create_image(130,130,anchor='center',image=img)
       #panel.configure(image=img)
       #panel.image = img
       tmp = filename.split("/")
       filename = tmp[len(tmp)-1]
       print(filename)
       tmp = filename.split("_")
       pic = tmp[len(tmp)-1].split(".")[0]
       print(pic)
       
   except:
       print('no file is loaded')

def maskDone():
   print("maskDone")
   cv2.imshow("mask_"+pic+".png", mouse.mask_img)
   cv2.imwrite("places356_mask/mask_"+pic+".png", mouse.mask_img)
   cv2.waitKey(0)

   
def saveFile():
   print("saveFile")
   image = "Places365_test_" + pic + ".jpg"
   mask = "mask_" + pic + ".png"
   output = "output_" + pic + ".png"
   Output(image,mask,output)
   #print("!python test.py --image \"drive/My Drive/generative_inpainting-master/examples/places356/Places365_test_" + pic + ".jpg\" --mask \"drive/My Drive/generative_inpainting-master/examples/places356/mask_" + pic + ".png\" --output \"drive/My Drive/generative_inpainting-master/examples/places356/output_" + pic + ".png\" --checkpoint \"drive/My Drive/generative_inpainting-master/model_logs/release_places2_256\"")
   

def callback(event):
    print("clicked at", event.x, event.y)

if __name__ == "__main__":
    window = tk.Tk()
    # 設定視窗標題、大小和背景顏色
    window.title('Image Inpainting')
    window.geometry('600x400')
    window.configure(background='white')
    
    titleFont = tkFont.Font(family="Microsoft JhengHei", size=20)
    buttonFont = tkFont.Font(family="Microsoft JhengHei", size=15)
    style = ttk.Style() 
    style.configure('TButton', font = ('calibri', 20, 'bold'), borderwidth = '4') 
    
    
    header_label = tk.Label(window, text='Image Inpainting',background='white',font=titleFont)
    header_label.pack()
    
    # 以下為 button_frame 群組
    button_frame = tk.Frame(window,background='white')
    button_frame.pack(side=tk.TOP)
    
    # 讀取新圖片
    read_file = tk.Button(button_frame, text ="讀檔", command = readFile,font=buttonFont,width=10)
    read_file.pack(side=tk.LEFT,padx=30,pady=20)
    
    # 畫框框
    mask_done = tk.Button(button_frame, text ="完成", command = maskDone,font=buttonFont,width=10)
    mask_done.pack(side=tk.LEFT,padx=30,pady=20)
    
    # 輸出結果
    save_file = tk.Button(button_frame, text ="儲存", command = saveFile,font=buttonFont,width=10)
    save_file.pack(side=tk.LEFT,padx=30,pady=20)
    
    # image視窗
    canvas = tk.Canvas(window, width=256, height=256,bg='black', cursor="cross")
    #canvas.bind("<Button-1>", callback)
    canvas.pack(side = "bottom",padx=30,pady=20)
    #panel = tk.Label(window,bg='gray')
    #panel.pack(side = "bottom", fill = "both", expand = "yes",padx=30,pady=30)
    
    # 運行主程式
    window.mainloop()