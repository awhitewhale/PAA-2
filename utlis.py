import cv2
import numpy as np
import random

def add_water_droplet(image, center, radius, refractive_index):
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    x_min, y_min = max(0, center[0] - radius), max(0, center[1] - radius)
    x_max, y_max = min(width, center[0] + radius), min(height, center[1] + radius)

    droplet_area = cv2.bitwise_and(image, image, mask=mask)
    droplet_area = droplet_area[y_min:y_max, x_min:x_max]

    map_x, map_y = np.meshgrid(np.linspace(x_min, x_max - 1, x_max - x_min, dtype=np.float32),
                               np.linspace(y_min, y_max - 1, y_max - y_min, dtype=np.float32))
    map_x = np.subtract(map_x, center[0])
    map_y = np.subtract(map_y, center[1])
    map_x = np.divide(map_x, radius)
    map_y = np.divide(map_y, radius)

    r_squared = np.add(np.square(map_x), np.square(map_y))
    r = np.sqrt(r_squared)
    r = np.where(r < 1, r, 0)

    theta = np.arcsin(r)
    theta_refracted = np.divide(theta, refractive_index)
    r_refracted = np.sin(theta_refracted)

    map_x = np.add(np.multiply(r_refracted, np.divide(map_x, r)), center[0])
    map_y = np.add(np.multiply(r_refracted, np.divide(map_y, r)), center[1])

    refracted_droplet = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    refracted_droplet = cv2.bitwise_and(refracted_droplet, refracted_droplet, mask=mask[y_min:y_max, x_min:x_max])

    output = image.copy()
    output[y_min:y_max, x_min:x_max] = cv2.addWeighted(image[y_min:y_max, x_min:x_max], 1, refracted_droplet, -1, 0)
    output[y_min:y_max, x_min:x_max] = cv2.add(output[y_min:y_max, x_min:x_max], refracted_droplet)

    return output

def add_random_water_droplets(image, num_droplets, min_radius, max_radius, refractive_index):
    height, width, _ = image.shape

    for _ in range(num_droplets):
        center = (random.randint(0, width), random.randint(0, height))
        radius = random.randint(min_radius, max_radius)
        image = add_water_droplet(image, center, radius, refractive_index)

    return image

if __name__ == "__main__":
    image = cv2.imread("553.png")
    refractive_index = 1.33  # 水的折射率

    output = add_random_water_droplets(image, num_droplets=20, min_radius=10, max_radius=30, refractive_index=refractive_index)

    cv2.imwrite("output.png", output)