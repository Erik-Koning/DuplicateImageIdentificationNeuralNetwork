import Tkinter as tk
import tkFileDialog
from PIL import ImageTk, Image

image = ""

# Asks the user for a file.
def get_file():
	filename = tkFileDialog.askopenfilename(filetypes=[("Images", "*.jpg")])
	w.config(text=filename)
	global image
	image = open_image(filename)
	imlabel.config(image=image)
	return filename

def open_image(path):
	return ImageTk.PhotoImage(Image.open(path))

# Initialize the Tk root widget.
root = tk.Tk()

# Create the label widget, as a child of the root widget.
w = tk.Label(root, fg="red")

# Pack allows sthe window to fit the size of the text.
w.pack()

imlabel = tk.Label(root)
imlabel.pack()

button = tk.Button(root, text='Open', width=25, command = get_file)
button.pack()

# Do the main GUI event loop.
root.mainloop()


