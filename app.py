"""
This code follows Nicholas Renotte's design
I did change some things to tailor it my computer
"""


import tkinter as tk
import customtkinter as ctk 

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import DiffusionPipeline 

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud") 
ctk.set_appearance_mode("dark") 

prompt = ctk.CTkEntry(master=app, height=40, width=512, text_color="black", fg_color="white") 
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512, text=None)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
# pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
# pipe.to(device)


pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()


def generate(): 
    with autocast(device): 
        _ = pipe(prompt.get(), num_inference_steps=1)
        image = pipe(prompt.get()).images[0] 
    
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img) 

trigger = ctk.CTkButton(master=app, height=40, width=120, text_color="white", fg_color="blue", command=generate) 
trigger.configure(text="Generate") 
trigger.place(x=206, y=60) 

app.mainloop()