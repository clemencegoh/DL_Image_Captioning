import tkinter
from PIL import Image, ImageTk
from tkinter import Tk, Frame, Button, Label, Message, Listbox, Scrollbar, Entry
from tkinter import filedialog, messagebox

import torch
import pickle
import os
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


class Captioner:
    def __init__(self,
                 encoder_path='model3/encoder-2-2000.ckpt',
                 decoder_path='model3/decoder-2-2000.ckpt',
                 vocab_path='data/vocab.pkl',
                 embed_size=256,
                 hidden_size=512,
                 num_layers=3):
        # Image preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        # Load vocabulary wrapper
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        # Build models
        encoder = EncoderCNN(embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        # Load the trained model parameters
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))

        self.vocab = vocab
        self.transforms = transform
        self.encoder = encoder
        self.decoder = decoder

    def CaptionImage(self, image):
        # Prepare an image
        image = load_image(image, self.transforms)
        image_tensor = image.to(device)

        # Generate an caption from the image
        feature = self.encoder(image_tensor)
        sampled_ids = self.decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        return sentence


class ModeController:
    def __init__(self, master):
        self.master = master

        self.activity_frame = Frame(root, bd=1)
        self.activity_frame.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=1)
        self.prediction_screen()

    def rebuild_activity(self):
        if self.activity_frame:
            self.activity_frame.destroy()
        self.activity_frame = Frame(root, bd=1)
        self.activity_frame.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=1)

    def prediction_screen(self):
        self.rebuild_activity()
        PredictImage(self.activity_frame)


class PredictImage:
    def __init__(self, master):
        self.master = master

        self.master.grid_columnconfigure(0, weight=6)
        self.master.grid_rowconfigure(0, weight=1)

        self.image = Label(self.master, text="Image")
        self.image.grid(row=0, column=0, sticky='nesw')

        self.browse = Button(self.master,
                             text="Browse",
                             command=self.read_image,
                             width=20,
                             height=3)
        self.browse.grid(row=1, column=0)

        # self.examples =

        self.results_txt = tkinter.StringVar()
        self.results_txt.set("Click browse to select an image and click predict to see predictions.")
        self.results = Label(self.master,
                             textvariable=self.results_txt,
                             anchor=tkinter.W,
                             width=30,
                             bd=1,
                             font="Times 12",
                             relief=tkinter.SUNKEN)
        self.results.grid(row=0, column=1, sticky='ns')
        # self.results.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        self.results.update_idletasks()
        self.results.configure(wraplength=self.results.winfo_width())

        self.predict_btn = Button(self.master, text="Predict", command=self.predict, width=20, height=3)
        self.predict_btn.grid(row=1, column=1)

        self.img = None
        self.path = ""

    def read_image(self):
        path = filedialog.askopenfilename(initialdir='.', title='Select file')
        if path:
            try:
                self.path = path
                img = Image.open(path)
                img.thumbnail((400, 300), Image.ANTIALIAS)
                self.img = img
                photo = ImageTk.PhotoImage(self.img)
                self.image.image = photo
                self.image.configure(image=photo)
            except OSError:
                messagebox.showerror("Error", "Failed to open file.")

    def predict(self):
        if not self.img:
            self.results_txt.set("No image.")
            return
        print(self.path)
        try:
            result = c.CaptionImage(self.path)
            self.results_txt.set(result)
        except Exception as e:
            self.results_txt.set("Error: {}".format(e))
            print(e)


if __name__ == '__main__':
    print("setting up tkinter...")
    root = Tk()
    root.geometry("1200x600")
    root.title("Image Captioning")
    controller = ModeController(root)

    # set default as prediction screen
    controller.prediction_screen()

    print('preparing model...')
    c = Captioner()

    print('completing gui setup...')
    root.mainloop()
