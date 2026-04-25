# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import cv2
import matplotlib.pyplot as plt

image = cv2.imread("zdjecie.jpg")

if image is None:
    print("Błąd: nie znaleziono zdjecia")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis("off")
plt.show()

height, width = image.shape[:2]
new_size = (width // 2, height // 2)

resized = cv2.resize(image, new_size)

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

plt.figure(figsize=(5,5))
plt.imshow(rotated, cmap="gray")
plt.title("(obrót: 90°, szary: 50%)")
plt.axis("off")
plt.show()

print("Macierz obrazu:")
print(rotated)
