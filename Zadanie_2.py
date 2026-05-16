import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("obraz.jpg")

if image is None:
    print("Błąd: nie znaleziono zdjecia")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.title("Obraz źródłowy")
plt.axis("off")
plt.show()

plt.figure(figsize=(8, 6))

colors = ('r', 'g', 'b')
labels = ('Red', 'Green', 'Blue')

for i, (col, label) in enumerate(zip(colors, labels)):
    plt.hist(
        image_rgb[:, :, i].ravel(), bins=256, color=col, alpha=0.5, label=label
    )

plt.title("Histogram RGB")
plt.xlabel("Intensywność")
plt.ylabel("Liczba pikseli")
plt.legend()
plt.show()

colors = ('r','g','b')
for i, col in enumerate(colors):
    plt.figure(figsize=(8,6))
    plt.hist(image_rgb[:,:,i].ravel(), bins=256, color=col)
    plt.title(f"Histogram kanału {col.upper()}")
    plt.xlabel("Intensywność")
    plt.ylabel("Ilość pikseli")
    plt.show()

hist_r = cv2.calcHist([image_rgb],[0],None,[256],[0,256]).flatten()
hist_g = cv2.calcHist([image_rgb],[1],None,[256],[0,256]).flatten()
hist_b = cv2.calcHist([image_rgb],[2],None,[256],[0,256]).flatten()

img_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()

total_pixels = image_rgb.shape[0] * image_rgb.shape[1]
pixels_dark = np.sum(hist_gray[0:6])
pixels_bright = np.sum(hist_gray[250:256])

pct_dark = pixels_dark / total_pixels
pct_bright = pixels_bright / total_pixels

std_dev = np.std(img_gray)

if pct_bright > 0.05 and pct_dark > 0.05:
    exposure_rating = "Zbyt wysoki kontrast"
elif pct_bright > 0.05:
    exposure_rating = "Zdjęcie prześwietlone"
elif pct_dark > 0.05:
    exposure_rating = "Zdjęcie niedoświetlone"
else:
    exposure_rating = "Ekspozycja poprawna"

if std_dev < 35:
    contrast_rating = "Niski kontrast"
elif 35 <= std_dev <= 75:
    contrast_rating = "Dobry, naturalny kontrast"
else:
    contrast_rating = "Wysoki, dynamiczny kontrast"

print("\nWyniki: ")
print(f"Procent niedoświetlonych pikseli (cienie): {pct_dark * 100:.2f}%")
print(f"Procent prześwietlonych pikseli (światła): {pct_bright * 100:.2f}%")
print(f"Wskaźnik kontrastu (Odchylenie standarowde): {std_dev:.2f} = {contrast_rating}")
print(f"Status ekspozycji: {exposure_rating}")

plt.figure(figsize=(8, 5))
plt.plot(hist_gray, color='black', lw=2)
plt.title(f"Histogram Jasności (Luminancja)\nOcena: {exposure_rating}")
plt.xlabel("Intensywność")
plt.ylabel("Ilość pikseli")
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)
plt.show()
