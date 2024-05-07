import cv2
import numpy as np
import matplotlib.pyplot as plt

def stereo_match(left_img, right_img, block_size=10, max_offset=50):

    height, width = left_img.shape



    half_block = block_size // 2
    disparity_map = np.zeros_like(left_img)

    for y in range(0, height -block_size):
        for x in range(0, width - block_size):
            best_offset = 0
            min_cost = float('inf')

            for offset in range(max_offset):
                left_block = left_img[y :y + block_size, x :x +block_size]
                right_block = right_img[y :y + block_size, x -offset:x +block_size- offset]

                if right_block.shape[1] != block_size:
                    continue

                cost = np.sum(np.square(left_block - right_block))  

                if cost < min_cost:
                    min_cost = cost
                    best_offset = offset

            disparity_map[y, x] = best_offset
    return disparity_map


left_img_color = cv2.imread('left.png')
right_img_color = cv2.imread('right.png')
left_img = cv2.cvtColor(left_img_color, cv2.COLOR_BGR2GRAY)
right_img = cv2.cvtColor(right_img_color, cv2.COLOR_BGR2GRAY)

h,w=left_img.shape
disparity_map = stereo_match(left_img, right_img, block_size=3, max_offset=32)
disparity_map_normalized = (disparity_map / disparity_map.max()) * 255#normalising the diaparity map
disparity_map_normalized = np.uint8(disparity_map_normalized)
image = np.zeros((h, w, 3), dtype=np.uint8)
image[:, :, 2] = disparity_map_normalized
image[:, :, 0] =255-image[:, :, 2]
plt.imshow(disparity_map_normalized, cmap='jet')
plt.colorbar()
plt.title('Depth map')
plt.axis('off')
plt.show()
cv2.imshow("depth map",image)
cv2.waitKey(0)
cv2.destroyAllWindows
