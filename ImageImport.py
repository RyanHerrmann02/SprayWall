import cv2
import numpy as np

path = ""

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


""" Definition "ENHANCECONTRAST" applies CLAHE (adaptive local contrast boosting) to a
    grayscale image before edge detection. This helps recover edges in areas where chalk
    dust has washed out contrast between the hold and background, since CLAHE boosts
    contrast locally rather than globally.
    gray        --> input grayscale image
    clip_limit  --> how aggressively to boost contrast (higher = more aggressive) """
def enhanceContrast(gray, clip_limit=1.25):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(10, 10))
    return clahe.apply(gray)


""" Definition "FILTERSMALLCONTOURS" to remove small edge contours (like bolts/hardware)
    from an edge map, keeping only contours above a minimum area. This works regardless
    of hold color/vibrancy since it filters purely by physical size, not saturation.
    edges       --> binary edge map (output of Canny)
    min_area    --> minimum contour area (in pixels) to keep; tune this against your
                    actual bolt size vs hold size in the image """
def filterSmallContours(edges, min_area):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(edges)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(filtered, [c], -1, 255, thickness=cv2.FILLED)
    return filtered


""" Definition "FINDWALLTOP" scans an edge map for long, near-horizontal lines using the
    Hough transform, and returns the y-coordinate of the topmost strong horizontal line
    found (assumed to be the top edge of the wall/ceiling boundary).
    edges           --> binary edge map
    min_line_length --> minimum pixel length to count as a real structural line, not noise
    max_angle_deg   --> maximum deviation from perfectly horizontal (in degrees) to accept """
def findWallTop(edges, min_line_length=600, max_angle_deg=2):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=min_line_length, maxLineGap=20)
    if lines is None:
        return None
    
    lines = lines.reshape(-1, 4)

    horizontal_ys = []
    for x1, y1, x2, y2 in lines:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= max_angle_deg:
            horizontal_ys.append(min(y1, y2))

    if not horizontal_ys:
        return None

    return min(horizontal_ys)  # topmost horizontal line


""" Definition "MASKABOVELINE" blacks out everything above a given y-coordinate in an image,
    used to remove ceiling/background clutter above the top of the wall.
    img     --> input image (any number of channels)
    y_line  --> y-coordinate; everything above this row will be zeroed out """
def maskAboveLine(img, y_line):
    result = img.copy()
    result[:y_line, ...] = 0
    return result


""" Definition "GRAYANDEDGEDETECT" to take the image that was inputted and then to convert
    it into a grayscale image and do an edge detection on it to create the outline of the holds
    this will later be implemented to define where hold are to layer the color on top of the
    gray image for clear route reading 
    img     --> input image to then manipulate color and grab edges """
def grayAndEdgeDetect(img):
    # Create gray image from colored image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Boost local contrast so chalk-covered areas still produce usable edges
    enhanced = enhanceContrast(gray)

    # Gaussian blur for edge detection
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Edge detect
    edges = cv2.Canny(blur, threshold1=100, threshold2=150)

    # Remove small contours (bolts/hardware), keep only larger hold-sized shapes
    kernel = np.ones((20, 20), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    holdEdges = filterSmallContours(closed, min_area=270)

    # Find the top-of-wall line and mask out everything above it
    wall_top_y = findWallTop(edges)
    if wall_top_y is not None:
        holdEdges = maskAboveLine(holdEdges, wall_top_y)
    else:
        print("No strong horizontal line found for wall top — skipping crop.")

    return [gray, edges, holdEdges]


if __name__ == "__main__":
    path = "SimpleRoutes.jpg"

    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Couldn't load image at {path}")

    filtered = grayAndEdgeDetect(image)
    cv2.imwrite("gray_layer.png", filtered[0])
    cv2.imwrite("edges_layer.png", filtered[1])
    cv2.imwrite("hold_edges_layer.png", filtered[2])

    show("Wall", image)
    show("Gray", filtered[0])
    show("Edge", filtered[1])
    show("Hold Edges", filtered[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()