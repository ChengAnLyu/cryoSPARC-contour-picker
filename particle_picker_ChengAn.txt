import pyautogui
import time
import tkinter as tk
from pynput import mouse

points = []
drawing = False
click_interval = 7  # density of picking

root = tk.Tk()
root.attributes('-topmost', True, '-fullscreen', True)
root.configure(bg='black')
root.wm_attributes('-transparentcolor', 'black')
canvas = tk.Canvas(root, bg='black', highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)

def on_click(x, y, button, pressed):

    global drawing, points

    if button == mouse.Button.left:
        if pressed:
            points = [(x, y)]
            drawing = True
        else:
            drawing = False
            if len(points) > 2:
                process_shape(points)
            canvas.delete("all")
            points.clear()

def on_move(x, y):
    if drawing:
        points.append((x, y))
        if len(points) > 1:
            x1, y1 = points[-2]
            x2, y2 = points[-1]
            canvas.create_line(x1, y1, x2, y2, fill="red", width=2)

def get_enclosed_points(points):
    x_min, y_min = min(p[0] for p in points), min(p[1] for p in points)
    x_max, y_max = max(p[0] for p in points), max(p[1] for p in points)

    enclosed_points = []
    for y in range(y_min, y_max, click_interval):
        for x in range(x_min, x_max, click_interval):
            if is_inside_polygon(x, y, points):
                enclosed_points.append((x, y))

    return enclosed_points

def is_inside_polygon(x, y, points):
    n = len(points)
    inside = False
    p1x, p1y = points[0]

    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            x_intersect = p1x + (y - p1y) * (p2x - p1x) / (p2y - p1y)
            if x <= x_intersect:
                inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def process_shape(points):
    click_points = get_enclosed_points(points)

    for px, py in click_points:
        pyautogui.click(px, py)
        time.sleep(0.01)

listener = mouse.Listener(on_click=on_click, on_move=on_move)
listener.start()

print("Hold down left button to draw...")
root.mainloop()
listener.join()
