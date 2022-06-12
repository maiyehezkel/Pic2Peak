
import tkinter as tk
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
# For measuring the inference time.
import time
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2




class Application(tk.Frame):
    def __init__(self,master):
        super().__init__(master)
        self.pack()
        self.master.geometry("642x640")
        self.master.configure(bg='white')
        self.master.title("Pic2Peak")
        
        self.create_widgets()

    def create_widgets(self):
        #Canvas
        self.canvas = tk.Canvas(self)
        self.canvas.configure(width=640, height=480, bg='white',highlightthickness=1, highlightbackground='black')
        self.canvas.pack(side=tk.TOP)
        
        #Logo
        logo = Image.open('logo.png')
        logo = ImageTk.PhotoImage(logo)
        logo_label = tk.Label(image=logo)
        logo_label.image = logo
        self.canvas.create_image(320,240,image=logo)

        #Menu
        menubar = Menu(self.master)
        menubar.configure(activebackground="grey")
        self.master.config(menu=menubar)
        menubar.add_command(label="Upload",command=self.loadImage)
        menubar.add_command(label="Start",command=self.startConvert)
        menubar.add_command(label="Clear", command=self.clearImage)
        menubar.add_command(label="Help", command=self.help)
        menubar.add_command(label="Quit", command=self.quit_app)

        #LogMessage
        self.messagebox = Listbox(self.master)
        self.messagebox.configure(width=640,height=250, background='black',fg='white', font=('Aerial 9 bold'))
        self.messagebox.pack(side=tk.BOTTOM)



    # Event Call Back
    def startConvert(self):
        module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
        detector = hub.load(module_handle).signatures['default']
        converted_img  = tf.image.convert_image_dtype(self.image_tk, tf.float32)[tf.newaxis, ...]
        start_time = time.time()
        result = detector(converted_img)
        end_time = time.time()

        result = {key:value.numpy() for key,value in result.items()}

        print("Found %d objects." % len(result["detection_scores"]))
        print("Inference time: ", end_time-start_time)


    
    def loadImage(self):
        self.filename = filedialog.askopenfilename(parent = root, title='Browse', filetypes=(('jpg files', '*.jpg'),('jpeg', '#.jpeg')))
        print(self.filename)
        try:
            self.image_bgr = cv2.imread(self.filename)
            self.height, self.width = self.image_bgr.shape[:2]
        except AttributeError:
            self.messagebox.insert(END,"The image could not be selected.\n Help: Please insert the image into the folder with Pick2pic")

        print(self.height, self.width)
        if self.width > self.height:
            self.new_size = (640,480)
        else:
            self.new_size = (480,480)

        self.image_bgr_resize = cv2.resize(self.image_bgr, self.new_size, interpolation=cv2.INTER_AREA)
        self.image_rgb = cv2.cvtColor( self.image_bgr_resize, cv2.COLOR_BGR2RGB )  #Since imread is BGR, it is converted to RGB
        self.image_PIL = Image.fromarray(self.image_rgb) #Convert from RGB to PIL format
        self.image_tk = ImageTk.PhotoImage(self.image_PIL) #Convert to ImageTk format
        self.image = self.canvas.create_image(320,240, image=self.image_tk)
     

    def help(self):
        helpPop = Toplevel(self.master)
        helpPop.geometry("350x500")
        helpPop.title("Help")
        Label(helpPop, text="help", font=('Mistral 18 bold')).place(x=150,y=80)

    def clearImage(self):
        try:
            self.canvas.delete(self.image)
        except AttributeError:
            self.messagebox.insert(END, "The canvas cannot be clear. Help: You need to insert a picture into Pic2Pick.")

    
    def quit_app(self):
        self.Msgbox = tk.messagebox.askquestion("Exit Applictaion", "Are you sure?", icon="warning")
        if self.Msgbox == "yes":
            self.master.destroy()




root = tk.Tk()
app = Application(master=root)#Inherit
app.mainloop()
