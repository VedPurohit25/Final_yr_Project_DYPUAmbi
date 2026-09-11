import os
import urllib.request

# Create images folder
os.makedirs('images', exist_ok=True)

# Sample fashion image URLs
sample_images = {
    "shoe1.jpg": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400",
    "shirt1.jpg": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=400",
    "jacket1.jpg": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400",
    "watch1.jpg": "https://images.unsplash.com/photo-1524805444758-089113d48a6d?w=400",
    "bag1.jpg": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400"
}

print("Downloading sample images...")
for filename, url in sample_images.items():
    filepath = os.path.join('images', filename)
    urllib.request.urlretrieve(url, filepath)
    print(f"Saved: {filepath}")

print("\nSample images downloaded successfully into 'images' folder!")
