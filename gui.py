import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from process import imageProcess  

class ImageProcessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Tool")
        self.img = None
        self.original_image_display = None
        self.processed_image_display = None

        # 创建主框架，将按钮和图像区域分开
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建左侧按钮区域
        button_frame = tk.Frame(main_frame)
        button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 创建右侧图像显示区域
        image_frame = tk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建选择文件按钮
        self.load_button = tk.Button(button_frame, text="Select Image File", command=self.load_image_file)
        self.load_button.pack(pady=5, fill=tk.X)

        # 创建灰度化按钮
        self.gray_button = tk.Button(button_frame, text="Grayscale", command=self.display_gray_image, state=tk.DISABLED)
        self.gray_button.pack(pady=5, fill=tk.X)

        # 创建直方图均衡化按钮
        self.equalize_button = tk.Button(button_frame, text="Histogram Equalization", command=self.display_equalized_image, state=tk.DISABLED)
        self.equalize_button.pack(pady=5, fill=tk.X)

        # 创建均值滤波按钮
        self.mean_filter_button = tk.Button(button_frame, text="Mean Filter", command=self.display_mean_filter_image, state=tk.DISABLED)
        self.mean_filter_button.pack(pady=5, fill=tk.X)

        # 创建中值滤波按钮
        self.median_filter_button = tk.Button(button_frame, text="Median Filter", command=self.display_median_filter_image, state=tk.DISABLED)
        self.median_filter_button.pack(pady=5, fill=tk.X)

        # 创建添加高斯噪声按钮
        self.gaussian_noise_button = tk.Button(button_frame, text="Add Gaussian Noise", command=self.display_gaussian_noise_image, state=tk.DISABLED)
        self.gaussian_noise_button.pack(pady=5, fill=tk.X)

        # 创建添加椒盐噪声按钮
        self.salt_pepper_noise_button = tk.Button(button_frame, text="Add Salt and Pepper Noise", command=self.display_salt_pepper_noise_image, state=tk.DISABLED)
        self.salt_pepper_noise_button.pack(pady=5, fill=tk.X)

        # 创建保存处理后图像按钮
        self.save_button = tk.Button(button_frame, text="Save Processed Image", command=self.save_processed_image, state=tk.DISABLED)
        self.save_button.pack(pady=5, fill=tk.X)

        # 创建作者信息标签
        self.author_label = tk.Label(root, text="Author: Heng Yongrui  Email: 22281067@bjtu.edu.cn", font=("Helvetica", 10, "italic"))
        self.author_label.pack(side=tk.BOTTOM, pady=10)


        # 创建原始图像显示标签和画布
        self.original_image_label = tk.Label(image_frame, text="Original Image")
        self.original_image_label.pack()
        self.original_image_canvas = tk.Label(image_frame)
        self.original_image_canvas.pack(pady=10)

        # 创建处理后图像显示标签和画布
        self.processed_image_label = tk.Label(image_frame, text="Processed Image")
        self.processed_image_label.pack()
        self.processed_image_canvas = tk.Label(image_frame)
        self.processed_image_canvas.pack(pady=10)

    def load_image_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("BMP Image", "*.bmp"), ("JPEG Image", "*.jpg"), ("PNG Image", "*.png")])
        # file_path = filedialog.askopenfilename()
        if not file_path:
            return
        try:
            self.img = imageProcess(file_path)
            self.display_original_image(self.img.rgb_image)
                
            self.gray_button.config(state=tk.NORMAL)
            self.equalize_button.config(state=tk.NORMAL)
            self.mean_filter_button.config(state=tk.NORMAL)
            self.median_filter_button.config(state=tk.NORMAL)
            self.gaussian_noise_button.config(state=tk.NORMAL)
            self.salt_pepper_noise_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Unable to load image file: {e}")

    def display_original_image(self, image_data):
        # 转换NumPy数组为PIL图像并显示
        image = Image.fromarray(image_data.astype('uint8'), 'RGB')
        image.thumbnail((256, 256))
        self.original_image_display = ImageTk.PhotoImage(image)
        self.original_image_canvas.config(image=self.original_image_display)

    def display_processed_image(self, image_data):
        # 转换NumPy数组为PIL图像并显示
        image = Image.fromarray(image_data.astype('uint8'), 'RGB')
        image.thumbnail((256, 256))
        self.processed_image_display = ImageTk.PhotoImage(image)
        self.processed_image_canvas.config(image=self.processed_image_display)
        self.save_button.config(state=tk.NORMAL)

    def display_gray_image(self):
        # 显示灰度化后的图像
        gray_image = self.img.gray_image
        gray_rgb = np.stack([gray_image] * 3, axis=-1)  # 将灰度图转换为伪RGB图像
        self.display_processed_image(gray_rgb)

    def display_equalized_image(self):
        # 进行直方图均衡化并显示
        equalized_image, _, _ = self.img.equalize(256)
        equalized_rgb = np.stack([equalized_image] * 3, axis=-1)  # 将均衡化灰度图转换为伪RGB图像
        self.display_processed_image(equalized_rgb)

    def display_mean_filter_image(self):
        # 进行均值滤波处理并显示
        mean_filtered_image = self.img.mean_filter(kernel_size=5)
        self.display_processed_image(mean_filtered_image)

    def display_median_filter_image(self):
        # 进行中值滤波处理并显示
        median_filtered_image = self.img.median_filter(kernel_size=5)
        self.display_processed_image(median_filtered_image)

    def display_gaussian_noise_image(self):
        # 添加高斯噪声并显示
        gaussian_noise_image = self.img.add_gaussian_noise(mean=0, sigma=15)
        self.display_processed_image(gaussian_noise_image)

    def display_salt_pepper_noise_image(self):
        # 添加椒盐噪声并显示
        salt_pepper_noise_image = self.img.add_salt_and_pepper_noise(salt_prob = 0.02, pepper_prob = 0.02)
        self.display_processed_image(salt_pepper_noise_image)
        
    def save_processed_image(self):
        # 保存均衡化图像
        save_path = filedialog.asksaveasfilename(filetypes=[("BMP Image", "*.bmp"), ("JPEG Image", "*.jpg"), ("PNG Image", "*.png")])
        if save_path:
            try:
                self.img.save_processed_img(save_path)
                messagebox.showinfo("Success", "Processed image saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Unable to save image: {e}")
                
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessGUI(root)
    root.mainloop()
