from PIL import Image
from io import BytesIO
import numpy as np

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

