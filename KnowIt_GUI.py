import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from keras.preprocessing.image import  img_to_array, load_img 
from keras.applications.vgg19 import  preprocess_input 

#loading the model
from keras.models import load_model
model = load_model('D:\Comp Science\KnowIt\custom_model.h5')



win = Tk()
win.geometry('950x530')
win.iconbitmap('D:\Comp Science\KnowIt\AllImages\icon.ico')
win.title('KnowIt - Classifying')
win.config(bg = "#2B7A0B")

#Setting background Image
bg = PhotoImage(file = "D:\Comp Science\KnowIt\AllImages\leaf-png-38610.png")
my_pic = Label(win, image = bg)
my_pic.place(x=0, y=0, relheight=1, relwidth=1)


Test1 = Label(win, text = 'Click "Open" to select image from memory and "Predict" to know  the class',
 bg='#2B7A0B', font=("Times_new_roman 14 bold")).pack()


global a

#Uploading a file
def openFile():
    global my_label
    filename = filedialog.askopenfilename(filetypes = (('JPG', '*.jpg'), ('All Files', '*.*')))
    path.config(text = filename)

    a = filename
    global file
    file = a
    
    my_img = ImageTk.PhotoImage(Image.open(filename))
    my_label = Label(image=my_img).pack()
    my_label.place(x=10, y=130)
    
    
#making predictions
def predicton():
    img = load_img(file, target_size = (270,270))
    i = img_to_array(img)
    im = preprocess_input(i)
    img = np.expand_dims(im, axis=0)
    pred = np.argmax(model.predict(img))
      

    if pred ==0:
        Msg =  ('Leaf is healthy')
    elif pred ==1:
        Msg= ( 'Leaf suffers "Bacterial Spot"')
    elif pred ==2:
        Msg= ( 'Leaf has "Early Blight"')
    elif pred ==3:
        Msg = ( 'Leaf has "Late Blight"')
    elif pred ==4:
        Msg = ( 'Leaf suffers "Leaf Mold"')
    elif pred ==5:
        Msg = ( 'Leaf is infected wuth "Septoria Leaf Spot"')
    elif pred ==6:
        Msg = ( 'Leaf suffers "Target Spot"')
    elif pred ==7:
        Msg = ( 'Leaf has "Two-Spotted Spider Mite"')
    elif pred ==8:
        Msg = ( 'Leaf has "Yellow Leaf Curl Virus"')
    elif pred ==9:
        Msg = ( 'Leaf has "Mosaic Virus"')    

    Messages.config(text = Msg)

#Display results    
global Messages 
Messages = Label(win,font = ("Calibri 20 bold"), bg ='#8DC142')
Messages.place(x=350, y=320 )

#Display path of image selected
global path 
path = Label(win,font = ("Calibri 8"), bg='#366620')
path.place(x=12,y=450)
global Test2
Test2 = Label(win, text="Directory Path of Image above", bg='#2B7A0B',
font =("Calibri 12") ).place(x=12, y = 470)


button1 = Button(text="Open", command = openFile, bg="#3EC70B", padx=10).place(x=150, y= 100)
button2 = Button(text="Predict", command = predicton, bg="#3EC70B", padx = 10).place(x=150, y= 300)

win.mainloop()
