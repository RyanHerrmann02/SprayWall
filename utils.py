import cv2

""" Definition "SHOW" to take inputted image and show it as a resized image to fit into the
    screen without changing the resolution may be used in another script
    name    --> Inputted name arbitrarily given by user
    img     --> Given image to scale (will change one needing grayscale)
    max_dim --> Maximum size that we want the window this is set to 1000
    scale   --> Will resize to be half, but if the scale is less than one the resolution
                will not be changed as we don't want to make the image too small if it already
                fits the screen"""
def show(name, img, max_dim=1000):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:  # only shrink, don't upscale small images
        display_img = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        display_img = img
    cv2.imshow(name, display_img)