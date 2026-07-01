import cv2
import matplotlib.pyplot as plt
import numpy as np
img_path = '../data/Stanford_Cars_dataset/test/Acura RL Sedan 2012/000132.jpg'


sigma=4
img_orig = cv2.imread(img_path)
img_gray = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
img_float = img_gray.astype(np.float32)
blurred = cv2.GaussianBlur(img_float, (17, 17), sigmaX=sigma, sigmaY=sigma)
hpf = img_float - blurred + 127
output = np.clip(hpf, 0, 255).astype(np.uint8)

plt.imshow(cv2.cvtColor(img_orig,cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()
plt.imshow(blurred,cmap='gray')
plt.title('Blurred Image')
plt.show()
plt.imshow(hpf,cmap='gray')
plt.title(f'HPF Image - sigma={sigma}')
plt.show()


