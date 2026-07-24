import cv2
import ImageImport as II
import utils

# Image input path
# path = "MacroSpray.jpg"
path = "SimpleRoutes.jpg"
image = cv2.imread(path)

# File not found exception
if image is None:
    raise FileNotFoundError(f"Couldn't load image at {path}")

# All import, filtering, and masking. Includes image in grayscale
grayscale, edges, fullMask = II.grayAndEdgeDetect(image)

utils.show("Wall", image)
utils.show("Edges", edges)
utils.show("Hold Mask", fullMask)
cv2.waitKey(0)
cv2.destroyAllWindows()