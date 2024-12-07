import numpy as np
from struct import unpack
import skimage
from PIL import Image

class bmpImage:
    """
    该类用于读取BMP图像文件

    Attributes:
        file_type (int): 文件类型标识符
        file_size (int): 文件大小
        reserved1 (int): 保留字段，必须为0
        reserved2 (int): 保留字段，必须为0
        offset_bits (int): 文件头到位图数据的偏移量
        header_size (int): 信息头的大小
        width (int): 图像的宽度（像素）
        height (int): 图像的高度（像素）
        planes (int): 颜色平面数
        bit_count (int): 每像素的比特数
        compression (int): 压缩类型
        image_size (int): 位图数据的大小
        x_pixels_per_meter (int): 水平分辨率
        y_pixels_per_meter (int): 垂直分辨率
        colors_used (int): 使用的调色板颜色数
        important_colors (int): 对显示有重要影响的调色板颜色数
        color_palette (np.ndarray): 调色板数据
        rgb_image (np.ndarray): 图像的RGB数据
        gray_image (np.ndarray): 图像的灰度数据
    """
    def __init__(self, file_path: str):
        with open(file_path, "rb") as file:
            self.file = file

            # 读取BMP文件头
            self.file_type = unpack("<H", file.read(2))[0]  # 文件类型
            self.file_size = unpack("<i", file.read(4))[0]  # 文件大小
            self.reserved1 = unpack("<H", file.read(2))[0]  # 保留字段，必须为0
            self.reserved2 = unpack("<H", file.read(2))[0]  # 保留字段，必须为0
            self.offset_bits = unpack("<i", file.read(4))[0]  # 从头到位图数据的偏移

            # 读取DIB信息头
            self.header_size = unpack("<i", file.read(4))[0]  # 信息头的大小
            self.width = unpack("<i", file.read(4))[0]  # 图像宽度（像素）
            self.height = unpack("<i", file.read(4))[0]  # 图像高度（像素）
            self.planes = unpack("<H", file.read(2))[0]  # 颜色平面数
            self.bit_count = unpack("<H", file.read(2))[0]  # 每像素的比特数
            self.compression = unpack("<i", file.read(4))[0]  # 压缩类型
            self.image_size = unpack("<i", file.read(4))[0]  # 位图数据的大小
            self.x_pixels_per_meter = unpack("<i", file.read(4))[0]  # 水平分辨率
            self.y_pixels_per_meter = unpack("<i", file.read(4))[0]  # 垂直分辨率
            self.colors_used = unpack("<i", file.read(4))[0]  # 使用的调色板颜色数
            self.important_colors = unpack("<i", file.read(4))[0]  # 对显示有重要影响的颜色数

            # 获取调色板和图像数据
            self.color_palette = self.get_color_palette()
            self.rgb_image = self.get_rgb_image()
            self.gray_image = self.get_gray_image()

    def get_color_palette(self) -> np.ndarray:
        """
        获取BMP图像的调色板

        Returns:
            np.ndarray: 调色板数据，包含颜色的RGB值
        """
        if self.offset_bits == 0x36:  # 16/24位图像没有调色板
            return None

        palette_size = 2 ** self.bit_count
        color_palette = np.zeros((palette_size, 3), dtype=np.int32)
        self.file.seek(0x36)

        for i in range(palette_size):
            blue = unpack("B", self.file.read(1))[0]
            green = unpack("B", self.file.read(1))[0]
            red = unpack("B", self.file.read(1))[0]
            alpha = unpack("B", self.file.read(1))[0]  # Alpha通道（忽略）
            color_palette[i] = [blue, green, red]

        return color_palette

    def get_rgb_values(self, pixel_data: str):
        """
        根据像素数据获取RGB值

        Args:
            pixel_data (str): 二进制的像素数据

        Returns:
            list: RGB颜色值
        """
        if len(pixel_data) <= 8:
            color_index = int(pixel_data, 2)
            return self.color_palette[color_index]
        elif len(pixel_data) == 16:
            blue = int(pixel_data[1:6], 2) * 8
            green = int(pixel_data[6:11], 2) * 8
            red = int(pixel_data[11:16], 2) * 8
            return [red, green, blue]
        elif len(pixel_data) == 24:
            blue = int(pixel_data[0:8], 2)
            green = int(pixel_data[8:16], 2)
            red = int(pixel_data[16:24], 2)
            return [red, green, blue]
        elif len(pixel_data) == 32:
            blue = int(pixel_data[0:8], 2)
            green = int(pixel_data[8:16], 2)
            red = int(pixel_data[16:24], 2)
            return [red, green, blue]

    def get_rgb_image(self) -> np.ndarray:
        """
        获取图像的RGB数据

        Returns:
            np.ndarray: RGB图像数据
        """
        self.height = abs(self.height)
        rgb_image = np.zeros((self.height, self.width, 3), dtype=np.int32)
        self.file.seek(self.offset_bits)

        for row in range(self.height):
            row_byte_count = ((self.width * self.bit_count + 31) >> 5) << 2
            row_bits = self.file.read(row_byte_count)
            row_bits = ''.join(format(byte, '08b') for byte in row_bits)

            for col in range(self.width):
                pixel_data = row_bits[col * self.bit_count: (col + 1) * self.bit_count]
                if self.height > 0:  # 图像倒立
                    rgb_image[self.height - 1 - row][col] = self.get_rgb_values(pixel_data)
                else:
                    rgb_image[row][col] = self.get_rgb_values(pixel_data)

        return rgb_image

    def get_gray_image(self) -> np.ndarray:
        """
        获取图像的灰度数据

        Returns:
            np.ndarray: 灰度图像数据
        """
        self.height = abs(self.height)
        gray_image = np.dot(self.rgb_image.reshape((self.height * self.width, 3)).astype(np.float32),
                            [0.299, 0.587, 0.114]).astype(np.int32)
        return gray_image.reshape((self.height, self.width))


class imageProcess():
    """
    该类实现各种图像处理操作，包括均值滤波、中值滤波、高斯噪声、椒盐噪声、直方图均衡化等
    """
    def __init__(self, file_path: str):
        if '.bmp' in file_path:
            # 读取BMP图像
            self.img = bmpImage(file_path)
            self.rgb_image = self.img.rgb_image
            self.gray_image = self.img.gray_image
        else:
            # 兼容其他格式的图像
            self.img = Image.open(file_path)
            self.rgb_image = np.array(self.img)
            self.gray_image = np.array(self.img.convert('L'))
            
        self.height, self.width, _ = self.rgb_image.shape

    def mean_filter(self, kernel_size: int = 3) -> np.ndarray:
        """
        对图像进行均值滤波

        Args:
            kernel_size (int): 滤波器的大小，默认为3

        Returns:
            np.ndarray: 滤波后的图像
        """
        padded_image = np.pad(self.rgb_image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)), mode='constant', constant_values=0)
        filtered_image = np.zeros_like(self.rgb_image)

        for row in range(self.height):
            for col in range(self.width):
                for channel in range(3):
                    filtered_image[row, col, channel] = np.mean(padded_image[row:row + kernel_size, col:col + kernel_size, channel])
        
        self.processed_image = filtered_image
        return filtered_image

    def median_filter(self, kernel_size: int = 3) -> np.ndarray:
        """
        对图像进行中值滤波

        Args:
            kernel_size (int): 滤波器的大小，默认为3

        Returns:
            np.ndarray: 滤波后的图像
        """
        padded_image = np.pad(self.rgb_image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)), mode='constant', constant_values=0)
        filtered_image = np.zeros_like(self.rgb_image)

        for row in range(self.height):
            for col in range(self.width):
                for channel in range(3):
                    filtered_image[row, col, channel] = np.median(padded_image[row:row + kernel_size, col:col + kernel_size, channel])

        self.processed_image = filtered_image
        return filtered_image

    def add_gaussian_noise(self, mean: float = 0, sigma: float = 25) -> np.ndarray:
        """
        为图像添加高斯噪声

        Args:
            mean (float): 噪声的均值，默认为0
            sigma (float): 噪声的标准差，默认为25

        Returns:
            np.ndarray: 添加噪声后的图像
        """
        gauss = np.random.normal(mean, sigma, self.rgb_image.shape).astype(np.int32)
        noisy_image = self.rgb_image + gauss
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.int32)
        
        self.processed_image = noisy_image
        return noisy_image
    
    def random_crop(self, crop_size: int = 256):
        """
        随机位置裁剪

        Args:
            crop_size (int): 裁剪大小，默认为256
        """
        start_x = np.random.randint(0, self.width - crop_size)
        start_y = np.random.randint(0, self.height - crop_size)
        cropped_image = self.rgb_image[start_y:start_y + crop_size, start_x:start_x + crop_size]

        self.processed_image = cropped_image
        return cropped_image
    
    def center_crop(self, crop_size: int = 256):
        """
        中心裁剪
        """
        crop_size = min(self.width // 2, self.height // 2, crop_size)
        start_x = (self.width - crop_size) // 2
        start_y = (self.height - crop_size) // 2
        cropped_image = self.rgb_image[start_y:start_y + crop_size, start_x:start_x + crop_size]

        self.processed_image = cropped_image
        return cropped_image
    
    def horizontal_flip(self):
        """
        随机水平翻转
        """
        flipped_image = np.fliplr(self.rgb_image)
        self.processed_image = flipped_image
        return flipped_image
    
    def vertical_flip(self):
        """
        随机垂直翻转
        """
        flipped_image = np.flipud(self.rgb_image)
        self.processed_image = flipped_image
        return flipped_image
    
    def rotation(self, angle: int = 90):
        """
        随机角度旋转

        Args:
            angle (int): 旋转角度，默认为90
        """
        # 将角度转换为弧度
        angle_rad = np.deg2rad(angle)
        
        # 获取图像的中心
        center_x, center_y = self.rgb_image.shape[1] // 2, self.rgb_image.shape[0] // 2
        
        # 创建旋转矩阵
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        # 创建空白图像
        rotated_image = np.zeros_like(self.rgb_image)
        
        # 遍历每个像素
        for y in range(self.rgb_image.shape[0]):
            for x in range(self.rgb_image.shape[1]):
                # 计算新位置
                new_x, new_y = np.dot(rotation_matrix, np.array([x - center_x, y - center_y]))
                new_x, new_y = int(new_x + center_x), int(new_y + center_y)
                
                # 检查新位置是否在图像范围内
                if 0 <= new_x < self.rgb_image.shape[1] and 0 <= new_y < self.rgb_image.shape[0]:
                    rotated_image[new_y, new_x] = self.rgb_image[y, x]
        
        self.processed_image = rotated_image
        return rotated_image
    
    
    def padding(self, padding_size: int = 256):
        """
        padding为正方形

        Args:
            padding_size (int): padding大小，默认为256
        """
        # 计算新的尺寸，确保图像是正方形
        new_size = max(self.height, self.width, padding_size)
        
        # 计算填充的大小
        pad_height = (new_size - self.height) // 2
        pad_width = (new_size - self.width) // 2
        
        # 创建一个新的图像，填充为黑色（0）
        padded_image = np.zeros((new_size, new_size, 3), dtype=self.rgb_image.dtype)
        
        # 将原始图像复制到新的图像中间
        padded_image[pad_height:pad_height + self.height, pad_width:pad_width + self.width] = self.rgb_image
        
        self.processed_image = padded_image
        return padded_image

    def add_salt_and_pepper_noise(self, salt_prob: float = 0.01, pepper_prob: float = 0.01) -> np.ndarray:
        """
        为图像添加椒盐噪声

        Args:
            salt_prob (float): 盐噪声的概率，默认为0.01
            pepper_prob (float): 椒噪声的概率，默认为0.01

        Returns:
            np.ndarray: 添加噪声后的图像
        """
        noisy_image = np.copy(self.rgb_image)
        num_salt = int(salt_prob * noisy_image.size / 3)
        num_pepper = int(pepper_prob * noisy_image.size / 3)

        # 添加盐噪声
        coords = [np.random.randint(0, i - 1, num_salt) for i in noisy_image.shape[:2]]
        noisy_image[coords[0], coords[1], :] = 255

        # 添加椒噪声
        coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy_image.shape[:2]]
        noisy_image[coords[0], coords[1], :] = 0

        self.processed_image = noisy_image
        return noisy_image

    def equalize(self, level: int=256):
        """
        对灰度图像进行直方图均衡化
        Args:
            level (int): 均衡化级别
        Returns:
            tuple: 均衡化后的图像、原始直方图、均衡化直方图
        """
        original_hist = np.zeros(256, dtype=np.int32)  # 原始直方图，用于统计每个灰度级别的像素数量
        max_value = self.gray_image.max()  # 灰度图像的最大值
        min_value = self.gray_image.min()  # 灰度图像的最小值
        interval = (max_value - min_value + 1) / level  # 将灰度范围划分为level个部分

        # 计算原始直方图
        for row in range(self.height):
            for col in range(self.width):
                original_hist[self.gray_image[row, col]] += 1

        # 计算均衡化直方图
        equalized_hist = np.zeros(level, dtype=np.float32)
        for i in range(level):
            # 将原始直方图的值按照划分后的区间进行累加
            equalized_hist[i] = np.sum(original_hist[min_value + int(i * interval): min_value + int((i + 1) * interval)])
        equalized_hist /= self.height * self.width  # 归一化处理，使直方图值在0到1之间

        # 计算累积概率分布函数(CDF)，并进行均衡化处理
        for i in range(1, level):
            equalized_hist[i] += equalized_hist[i - 1]
        equalized_hist *= level  # 将累积概率映射到新的灰度级别
        equalized_hist = np.around(equalized_hist)  # 四舍五入取整
        equalized_hist /= level  # 再次归一化处理
        equalized_hist = np.floor(equalized_hist * 255).astype(np.int32)  # 将均衡化后的值映射回0到255的范围

        # 生成均衡化后的图像和直方图
        self.processed_image = np.zeros_like(self.gray_image)  # 均衡化后的灰度图像
        self.equalized_hist = np.zeros(256, dtype=np.int32)  # 均衡化后的直方图

        for row in range(self.height):
            for col in range(self.width):
                # 将原始灰度值映射为均衡化后的灰度值
                self.processed_image[row, col] = equalized_hist[int((self.gray_image[row, col] - min_value) / interval)]
                self.equalized_hist[self.processed_image[row, col]] += 1

        return self.processed_image, original_hist, self.equalized_hist

    def save_processed_img(self, save_path: str):
        """
        保存处理后的图像

        Args:
            save_path (str): 保存路径
        """
        self.save_img(image=self.processed_image, save_path=save_path)

    def save_img(self, image: np.ndarray, save_path: str):
        """
        保存图像为BMP格式。

        Args:
            image (np.ndarray): 图像数据。
            save_path (str): 保存路径。
        """
        # 使用skimage.io.imsave进行保存
        skimage.io.imsave(save_path, image.astype(np.uint8))
        


        
    # def save_img(self, image: np.ndarray, save_path: str):
    #     """
    #     保存BMP图像

    #     Args:
    #         image (np.ndarray): 图像数据
    #         save_path (str): 保存路径
    #     """
    #     with open(save_path, "wb") as file:
    #         # 写入BMP文件头
    #         file.write(int(self.file_type).to_bytes(2, byteorder='little'))  # 文件类型
    #         file.write(int(0x36 + 0x100 * 4 + self.width * abs(self.height)).to_bytes(4, byteorder='little'))  # 文件大小
    #         file.write(int(0).to_bytes(4, byteorder='little'))  # 保留字段，必须设置为0
    #         file.write(int(0x36 + 0x100 * 4).to_bytes(4, byteorder='little'))  # 文件头到位图数据的偏移

    #         # 写入DIB信息头
    #         file.write(int(40).to_bytes(4, byteorder='little'))  # 信息头的大小
    #         file.write(int(self.width).to_bytes(4, byteorder='little'))  # 图像宽度
    #         file.write(int(self.height).to_bytes(4, byteorder='little'))  # 图像高度
    #         file.write(int(self.planes).to_bytes(2, byteorder='little'))  # 颜色平面数
    #         file.write(int(8).to_bytes(2, byteorder='little'))  # 每像素的比特数
    #         file.write(int(self.compression).to_bytes(4, byteorder='little'))  # 压缩类型
    #         file.write(int(self.image_size).to_bytes(4, byteorder='little'))  # 位图数据大小
    #         file.write(int(self.x_pixels_per_meter).to_bytes(4, byteorder='little'))  # 水平分辨率
    #         file.write(int(self.y_pixels_per_meter).to_bytes(4, byteorder='little'))  # 垂直分辨率
    #         file.write(int(0x100 * 4).to_bytes(4, byteorder='little'))  # 使用的调色板颜色数
    #         file.write(int(0).to_bytes(4, byteorder='little'))  # 对显示有重要影响的颜色数

    #         # 写入调色板
    #         for i in range(256):
    #             file.write(int(i).to_bytes(1, byteorder='little'))
    #             file.write(int(i).to_bytes(1, byteorder='little'))
    #             file.write(int(i).to_bytes(1, byteorder='little'))
    #             file.write(int(0).to_bytes(1, byteorder='little'))

    #         # 写入图像数据
    #         for row in range(abs(self.height)):
    #             for col in range(self.width):
    #                 if self.height > 0:
    #                     file.write(int(image[self.height - 1 - row][col]).to_bytes(1, byteorder='little'))
    #                 else:
    #                     file.write(int(image[row][col]).to_bytes(1, byteorder='little'))
    #             file.write(b'0' * ((((self.width * 8 + 31) >> 5) << 2) - 8 * self.width))
    #         # # 写入图像数据
    #         # for row in range(abs(self.height)):
    #         #     for col in range(self.width):
    #         #         pixel = image[row, col]
    #         #         file.write(int(pixel[2]).to_bytes(1, byteorder='little'))  # 写入蓝色通道
    #         #         file.write(int(pixel[1]).to_bytes(1, byteorder='little'))  # 写入绿色通道
    #         #         file.write(int(pixel[0]).to_bytes(1, byteorder='little'))  # 写入红色通道

    #         #         # 每行补齐到4字节的倍数
    #         #         row_padding = (4 - (self.width * 3) % 4) % 4
    #         #         file.write(b'\x00' * row_padding)
