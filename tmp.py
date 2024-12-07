from PIL import Image

jpg_file = './test/dark.jpg'  
bmp_file = './test/dark.bmp'  
img = Image.open(jpg_file)
img.save(bmp_file, 'BMP')