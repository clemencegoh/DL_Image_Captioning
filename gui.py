import tkinter
from PIL import Image, ImageTk
from tkinter import Tk, Frame, Button, Label, Message, Listbox, Scrollbar, Entry
from tkinter import filedialog, messagebox

import os
import torch
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models


class ModeController:
    def __init__(self, master):
        self.master = master

        self.mode_frame = Frame(root, bd=1)
        self.mode_frame.pack(side=tkinter.TOP, fill=tkinter.X)

        # self.label = Label(self.mode_frame, text="Image Captioning")
        # self.label.pack(side=tkinter.LEFT)

        self.predict = Button(self.mode_frame, text="Predict Image", command=self.prediction_screen)
        self.predict.pack(side=tkinter.LEFT)

        # self.view = Button(self.mode_frame, text="View precomputed predictions", command=self.view_screen)
        # self.view.pack(side=tkinter.LEFT)

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

    def view_screen(self):
        self.rebuild_activity()
        ViewResults(self.activity_frame)


class PredictImage:
    def __init__(self, master):
        self.master = master
        self.master.grid_columnconfigure(0, weight=6)
        self.master.grid_rowconfigure(0, weight=1)

        self.image = Label(self.master, text="Image")
        self.image.grid(row=0, column=0, sticky='nesw')

        self.browse = Button(self.master, text="Browse", command=self.read_image, width=20, height=3)
        self.browse.grid(row=1, column=0)

        # self.results_txt = tkinter.StringVar()
        # self.results = Message(self.master, textvariable=self.results_txt, relief=tkinter.SUNKEN, anchor='nw')
        # self.results.bind("<Configure>", lambda e: self.results.configure(width=e.width-10))
        # self.results_txt.set("Results")
        # self.results.grid(row=0, column=1, sticky='nesw')

        self.results_txt = tkinter.StringVar()
        self.results = Label(self.master, textvariable=self.results_txt, anchor='nw', width=30, justify=tkinter.LEFT, bd=1, relief=tkinter.SUNKEN)
        self.results_txt.set("Click browse to select an image and click predict to see predictions.")
        self.results.grid(row=0, column=1, sticky='ns')
        # self.results.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        self.results.update_idletasks()
        self.results.configure(wraplength=self.results.winfo_width())

        self.predict_btn = Button(self.master, text="Predict", command=self.predict, width=20, height=3)
        self.predict_btn.grid(row=1, column=1)

        self.img = None

    def read_image(self):
        path = filedialog.askopenfilename(initialdir='.', title='Select file')
        if path:
            try:
                self.img = Image.open(path).convert("RGB")
                photo = ImageTk.PhotoImage(self.img)
                self.image.image = photo
                self.image.configure(image=photo)
            except OSError:
                messagebox.showerror("Error", "Failed to open file.")

    def predict(self):
        if not self.img:
            self.results_txt.set("No image.")
            return
        try:
            model
        except NameError:
            self.results_txt.set("No model. Check path set in script.")
            return

        model.eval()
        with torch.no_grad():
            t_img = transform(self.img)
            output = model(t_img.unsqueeze(0))
            output = output.squeeze(0)
            output = torch.sigmoid(output)
            output = output.data.tolist()

            output = sorted(((val, idx) for idx, val in enumerate(output)), reverse=True)
            
            msg = ""
            for val, idx in output:
                msg += f"{IMAGE_SET[idx]}: {val:.6f}\n"
            self.results_txt.set(msg)


class ViewResults():
    def __init__(self, master):
        self.master = master

        self.list = Listbox(self.master, selectmode=tkinter.SINGLE, exportselection=False)
        self.categories = val_results.columns[1:21]
        self.filenames = val_results['filename']
        for cat in self.categories:
            self.list.insert(tkinter.END, cat)
        self.list.pack(side=tkinter.LEFT, fill=tkinter.Y)
        self.list.bind("<ButtonRelease-1>", lambda e: self.show_selection())

        self.thumbnail_frame = Frame(self.master)  # dummy frame

    def show_selection(self):
        idx = self.list.curselection()[0]
        cat = self.categories[idx]
        values = val_results[cat]
        sorted_values = values.sort_values(ascending=False)
        sorted_filenames = self.filenames[sorted_values.index]

        self.thumbnail_frame.destroy()
        self.thumbnail_frame = Frame(self.master, bd=1, relief=tkinter.SUNKEN)
        self.thumbnail_frame.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=1)
        self.thumbnails = Thumbnails(self.thumbnail_frame, sorted_filenames, cat)


class Thumbnails():
    def __init__(self, master, sorted_filenames, category):
        self.master = master
        self.filenames = sorted_filenames
        self.category = category
        self.cur_page_num = 0
        self.images = []
        self.max_page = int(len(self.filenames) / 20) + 1

        self.results_txt = tkinter.StringVar()
        self.results = Label(self.master, textvariable=self.results_txt, anchor='nw', width=30, justify=tkinter.LEFT, bd=1, relief=tkinter.SUNKEN)
        self.results_txt.set("Click on an image to show results.")
        self.results.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        self.results.update_idletasks()
        self.results.configure(wraplength=self.results.winfo_width())

        self.left_frame = Frame(self.master, bd=1)
        self.left_frame.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=1)

        self.thumbnail_frame = Frame(self.left_frame)  # dummy frame
        self.thumbnail_frame.pack(fill=tkinter.BOTH, expand=1)

        btn_frame = Frame(self.left_frame)
        btn_frame.pack(side=tkinter.BOTTOM, fill=tkinter.X)

        self.next_btn = Button(btn_frame, text="Next", command=self.next)
        self.next_btn.pack(side=tkinter.RIGHT)

        self.prev_btn = Button(btn_frame, text="Previous", command=self.prev)
        self.prev_btn.pack(side=tkinter.RIGHT)

        label = Label(btn_frame, text="Page: ")
        label.pack(side=tkinter.LEFT)

        self.page = Entry(btn_frame, width=4)
        self.page.bind("<Return>", self.skip)
        self.page.bind("<KP_Enter>", self.skip)
        self.page.pack(side=tkinter.LEFT)

        label = Label(btn_frame, text=f" of {self.max_page}")
        label.pack(side=tkinter.LEFT)

        self.show()

    def skip(self, e):
        try:
            page = int(e.widget.get())
        except ValueError:
            return

        if page < 1:
            page = 0
        elif page > self.max_page:
            page = self.max_page - 1
        else:
            page -= 1

        if page != self.cur_page_num:
            self.cur_page_num = page
            self.show()

    def show(self):
        self.thumbnail_frame.destroy()
        self.thumbnail_frame = Frame(self.left_frame)
        self.thumbnail_frame.pack(fill=tkinter.BOTH, expand=1)
        self.thumbnail_frame.update_idletasks()

        self.page.delete(0, tkinter.END)
        self.page.insert(0, str(self.cur_page_num + 1))

        per_row = 5
        per_col = 4
        width = int((self.thumbnail_frame.winfo_width() - 20) / per_row)
        height = int((self.thumbnail_frame.winfo_height() - 20) / per_col)

        start_idx = self.cur_page_num * 20
        end_idx = start_idx + 20
        end_idx = end_idx if end_idx < len(self.filenames) else len(self.filenames) - 1

        to_show = self.filenames[start_idx:end_idx]
        for idx, entry in enumerate(to_show.items()):
            row = int(idx / per_row)
            col = idx % per_row

            dataframe_index = entry[0]
            fn = entry[1]

            path = os.path.join(IMAGE_DIR, fn + '.jpg')

            img = Image.open(path).convert("RGB")
            img.thumbnail((width, height))
            photo = ImageTk.PhotoImage(img)

            image = Label(self.thumbnail_frame, image=photo)
            image.image = photo
            image.idx = dataframe_index
            image.bind('<ButtonRelease-1>', self.populate_results)
            self.images.append(image)
            image.grid(row=row, column=col, sticky='nsew')

        self._redraw = None
        self.thumbnail_frame.bind("<Configure>", self.redraw)

    def redraw(self, event):
        if self._redraw:
            self.thumbnail_frame.after_cancel(self._redraw)
        self._redraw = self.thumbnail_frame.after(500, self.show)

    def populate_results(self, e):
        idx = e.widget.idx
        predictions = results.loc[idx]
        ground_truth = gt.loc[idx]
        
        predictions.sort_values(ascending=False, inplace=True)
        ground_truth.sort_values(ascending=False, inplace=True)

        msg = str(predictions) + "\n\n\n" + str(ground_truth)
        index_of_category = msg.find(self.category)
        msg = msg[:index_of_category] + "-> " + msg[index_of_category:]
        index_of_gt = msg.find("gt" + self.category)
        msg = msg[:index_of_gt] + "-> " + msg[index_of_gt:]
        self.results_txt.set(msg)

    def next(self):
        if self.cur_page_num < self.max_page - 1:
            self.cur_page_num += 1
            self.show()

    def prev(self):
        if self.cur_page_num > 0:
            self.cur_page_num -= 1
            self.show()


# class FiveCropResnet(models.resnet.ResNet):
#     def forward(self, x):
#         size = list(x.size())
#         model_size = [-1] + size[2:]
#         output_size = size[:2] + [-1]
#         x = x.view(model_size)
#         x = super().forward(x)
#         x = x.view(output_size)
#         x = x.mean(dim=1)
#         return x


def get_model(path_to_parameters):
    model = models.resnet18(pretrained=False)
    inp_feature_count = model.fc.in_features
    model.fc = nn.Linear(inp_feature_count, 20)
    model.load_state_dict(torch.load(path_to_parameters, map_location=torch.device('cpu')))
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



if __name__ == '__main__':
    # Set variables here
    # IMAGE_SET = [
    #     'aeroplane', 'bicycle', 'bird', 'boat',
    #     'bottle', 'bus', 'car', 'cat', 'chair',
    #     'cow', 'diningtable', 'dog', 'horse',
    #     'motorbike', 'person', 'pottedplant',
    #     'sheep', 'sofa', 'train',
    #     'tvmonitor']
    # IMAGE_DIR = "pascalvoc/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
    MODEL_PATH = './model.pt'
    VAL_RESULTS_PATH = './val_results.csv'

    # model = get_model(MODEL_PATH)
    # transform = get_transform()
    # val_results = pd.read_csv(VAL_RESULTS_PATH)
    # results = val_results[val_results.columns[1:21]]
    # gt = val_results[val_results.columns[21:]]

    root = Tk()
    root.geometry("1200x600")
    root.title("Image Captioning")
    controller = ModeController(root)

    root.mainloop()
