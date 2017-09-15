import sys
from PIL import Image

def main():
	ImageFile = sys.argv[1]

	im = Image.open(ImageFile)
	width, height = im.size
	result = Image.new("RGB",(width,height))
	for i in range(0,width):
		for j in range(0,height):
			r,g,b = im.getpixel((i,j))
			r = r//2
			g = g//2
			b = b//2
			pix = (r,g,b)
			result.putpixel((i,j),pix)
	result.save("Q2.png")

if __name__ == "__main__":
	main()