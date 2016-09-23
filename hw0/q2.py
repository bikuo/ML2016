import sys
from PIL import Image
file = sys.argv[1]
im = Image.open(file)
im2 = im.rotate(180)
#im2.show()
im2.save("ans2.png","png")
