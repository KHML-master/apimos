import tkinter.filedialog
import tkinter as tk

from PIL import Image, ImageTk, ImageDraw
import os


class PickingPoints(tk.Frame):
    def __init__(self, parent, input_img_path, mask_template_dir, mask_name, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.dots = []
        self.current_dots = []
        self.masks = []
        self.dot_width = 5
        self.padding = 200
        self.current_mask = -1
        self.show_mask_active = True
        self.mask_template_dir = mask_template_dir
        self.mask_name = mask_name

        # Image
        img = Image.open(input_img_path)
        self.im_width, self.im_height = img.size
        self.img = ImageTk.PhotoImage(img)
        self.parent.one = self.img  # Prevent Garbage Collection

        # Canvas
        self.canvas = tk.Canvas(width=self.im_width+self.padding, height=self.im_height+self.padding)
        self.canvas.pack(side='top', fill='both', expand=True)
        self.canvas.create_image(((self.im_width/2) + self.padding/2, (self.im_height/2) + self.padding/2), image=self.img, anchor='center')

        # Mask Background
        self.mask_bg = self.canvas.create_rectangle(self.padding/2, self.padding/2, self.im_width + self.padding/2, self.im_height + self.padding/2, fill='black')
        self.canvas.itemconfigure(self.mask_bg, state=tk.HIDDEN)

        # Bindings
        self.canvas.bind('<Button-1>', self.draw_mask)  # Left Mouse Press
        self.canvas.bind('<Button-3>', self.complete_drawing_mask)  # Right Mouse Press

        self.parent.bind('m', self.show_mask)
        self.parent.bind('<Return>', self.output_mask)

    def draw_mask(self, event):
        print('Drawing Mask')

        # Position of mouse press
        x1, y1 = (event.x - self.dot_width), (event.y - self.dot_width)
        x2, y2 = (event.x + self.dot_width), (event.y + self.dot_width)

        # Dot Creation
        dot = self.canvas.create_oval(x1, y1, x2, y2, fill='red')  # Draw Dot
        self.current_dots.append(dot)

        # Refresh Mask
        self.canvas.delete(self.current_mask)
        dot_coords = self.get_dot_coordinates(self.current_dots, self.dot_width)
        self.current_mask = self.canvas.create_polygon(dot_coords, fill='white')  # Draw Polygon

    def complete_drawing_mask(self, event):
        print('Completed Mask')

        # Append mask to mask list
        dot_coords = self.get_dot_coordinates(self.current_dots, self.dot_width)
        mask = self.canvas.create_polygon(dot_coords, fill='white')  # Draw Polygon
        self.masks.append(mask)

        # Hide Current Mask
        self.canvas.itemconfigure(self.current_mask, state=tk.HIDDEN)

        # Delete Current Dots
        for d in self.current_dots:
            self.canvas.delete(d)

        # Refresh
        self.current_dots = []

    def show_mask(self, event):
        if self.show_mask_active is False:
            self.show_mask_active = True
            print('Show masks')

            # Show Mask Background
            self.canvas.itemconfigure(self.mask_bg, state=tk.NORMAL)

            # Show ROI Masks
            for m in self.masks:
                self.canvas.itemconfigure(m, state=tk.NORMAL)
        else:
            self.show_mask_active = False
            print('Hide masks')

            # Hide Mask Background
            self.canvas.itemconfigure(self.mask_bg, state=tk.HIDDEN)

            # Hide ROI Masks
            for m in self.masks:
                self.canvas.itemconfigure(m, state=tk.HIDDEN)

    def output_mask(self, event):
        print('extract mask')
        print('Done?')  # Maybe add confirmation

        # Show Mask Background
        self.canvas.itemconfigure(self.mask_bg, state=tk.NORMAL)

        # Show ROI Masks
        for m in self.masks:
            self.canvas.itemconfigure(m, state=tk.NORMAL)

        # Complete
        output_mask = Image.new('RGB', (self.im_width, self.im_height), (0, 0, 0))
        draw = ImageDraw.Draw(output_mask)

        # Draw each mask to output mask
        for m in self.masks:
            true_coords = [x - self.padding/2 for x in self.canvas.coords(m)]
            draw.polygon(true_coords, (255, 255, 255))

        # Saving Mask
        output_mask.save(self.mask_name)
        self.parent.destroy()

    def get_dot_coordinates(self, dots, dot_width):
        dot_coords = []
        for d in dots:
            x1, y1, _, _ = self.canvas.coords(d)
            x1 = int(x1 + dot_width)
            y1 = int(y1 + dot_width)
            x, y = self.limit_coordinates(x1, y1)
            dot_coords.append([x, y])
        return dot_coords

    def limit_coordinates(self, x, y):
        # y within border
        if y > self.im_height + (self.padding / 2):
            y = self.im_height + (self.padding / 2)
        elif y < 0 + (self.padding/2):
            y = 0 + self.padding/2

        # x within border
        if x > self.im_width + (self.padding / 2):
            x = self.im_width + (self.padding / 2)
        elif x < 0 + (self.padding / 2):
            x = 0 + self.padding / 2
        return x, y


def create_mask(dataset_dir, input_image_path, directory_path):
    # Mask Directory of Mask templates
    mask_template_dir = f'{dataset_dir}/masks'
    if os.path.isdir(mask_template_dir) is False:
        os.mkdir(mask_template_dir)

    # Mask Name
    mask_path = f'{mask_template_dir}/{os.path.basename(directory_path)}_mask.jpg'

    # TK
    root = tk.Tk()
    PickingPoints(root, input_image_path, mask_template_dir, mask_path).pack(side='top', fill='both', expand=True)
    root.mainloop()

    return mask_path
