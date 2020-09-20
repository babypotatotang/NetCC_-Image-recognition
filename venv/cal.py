import PIL.Image as pilimg
import numpy as np
import os
# Read image

im = pilimg.open( os.path.join('pic.jpg',file ))
im.show()
pix=np.array(im)
path="C:\NetCC\NetCC_-Image-recognition"

files=os.listdir(C:\NetCC\NetCC_-Image-recognition)
