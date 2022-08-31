import numpy as np
import cv2
import matplotlib.pyplot as plt


img_near = cv2.imread('res19-near.jpg', cv2.IMREAD_COLOR)
img_far = cv2.imread('res20-far.jpg', cv2.IMREAD_COLOR)

img_near = cv2.cvtColor(img_near, cv2.COLOR_BGR2RGB)
img_far = cv2.cvtColor(img_far, cv2.COLOR_BGR2RGB)

# near_left_eye = np.array([455, 235])
# far_left_eye = np.array([467, 203])

near_eyes = np.array([[455, 234], [455, 416]])  # dist = 182
far_eyes = np.array([[467, 203], [467, 387]])  # dist = 184

near_shifted = img_near[:, near_eyes[0, 1] - far_eyes[0, 1]:]
far_shifted = img_far[far_eyes[0, 0] - near_eyes[0, 0]:, :]

near_reshaped = near_shifted[:-(near_shifted.shape[0] - far_shifted.shape[0]), :]
far_reshaped = far_shifted[:, :-(far_shifted.shape[1] - near_shifted.shape[1])]

###########################################################
# Saving aligned images####################################
###########################################################
plt.imsave('res21-near.jpg', near_reshaped)
plt.imsave('res22-far.jpg', far_reshaped)
###########################################################

near = near_reshaped.copy()
far = far_reshaped.copy()

near_dft = np.zeros(near.shape, dtype=np.complex128)
far_dft = np.zeros(far.shape, dtype=np.complex128)

for i in range(3):
    near_dft[:, :, i] = np.fft.fftshift(np.fft.fft2(near[:, :, i]))
    far_dft[:, :, i] = np.fft.fftshift(np.fft.fft2(far[:, :, i]))

near_dft_log_abs = np.log(np.abs(near_dft))
near_dft_log_abs = near_dft_log_abs / near_dft_log_abs.max() * 255
far_dft_log_abs = np.log(np.abs(far_dft))
far_dft_log_abs = far_dft_log_abs / far_dft_log_abs.max() * 255
# Saving aligned images
plt.imsave('res23-dft-near.jpg', near_dft_log_abs.astype(np.uint8))
plt.imsave('res24-dft-far.jpg', far_dft_log_abs.astype(np.uint8))
# Filters
s = 16  # lpf
r = 25   # hpf

x_axis = np.arange(near_dft.shape[0]) - near_dft.shape[0] // 2
y_axis = np.arange(near_dft.shape[1]) - near_dft.shape[1] // 2
x_filter = np.exp(-np.square(x_axis) / 2 / r ** 2).reshape((-1, 1))
y_filter = np.exp(-np.square(y_axis) / 2 / r ** 2).reshape((1, -1))
lpf_ = x_filter @ y_filter
hpf = 1 - lpf_

x_axis = np.arange(far_dft.shape[0]) - far_dft.shape[0] // 2
y_axis = np.arange(far_dft.shape[1]) - far_dft.shape[1] // 2
x_filter = np.exp(-np.square(x_axis) / 2 / s ** 2).reshape((-1, 1))
y_filter = np.exp(-np.square(y_axis) / 2 / s ** 2).reshape((1, -1))
lpf = x_filter @ y_filter

hpf_display = hpf / hpf.max() * 255
hpf_display = hpf_display.astype(np.uint8)
lpf_display = lpf / lpf.max() * 255
lpf_display = lpf_display.astype(np.uint8)

plt.imsave('res25-highpass-' + str(r) + '.jpg', hpf_display, cmap='gray')
plt.imsave('res26-lowpass-' + str(s) + '.jpg', lpf_display, cmap='gray')

# This is necessary.
near_dft = near_dft / near_dft.max() * 255
far_dft = far_dft / far_dft.max() * 255

high_passed = np.zeros(near_dft.shape, dtype=np.complex128)
low_passed = np.zeros(far_dft.shape, dtype=np.complex128)
for i in range(3):
    high_passed[:, :, i] = hpf * near_dft[:, :, i]
    low_passed[:, :, i] = lpf * far_dft[:, :, i]

# high_passed = high_passed / high_passed.max() * 255
# low_passed = low_passed / low_passed.max() * 255

high_passed_abs = np.abs(high_passed)
high_passed_abs = high_passed_abs / high_passed_abs.max() * 255
high_passed_abs = np.log(high_passed_abs + 1)
high_passed_abs = high_passed_abs / high_passed_abs.max() * 255
low_passed_abs = np.abs(low_passed)
low_passed_abs = low_passed_abs / low_passed_abs.max() * 255
low_passed_abs = np.log(low_passed_abs + 1)
low_passed_abs = low_passed_abs / low_passed_abs.max() * 255

plt.imsave('res27-highpassed.jpg', high_passed_abs.astype(np.uint8))
plt.imsave('res28-highpassed.jpg', low_passed_abs.astype(np.uint8))


out_dft = np.zeros(near_dft.shape, dtype=np.complex128)
out = np.zeros(near_dft.shape, dtype=np.float64)
for i in range(3):
    out_dft[:, :, i] = high_passed[:, :, i] * 0.55 + low_passed[:, :, i] * 0.45
    out[:, :, i] = np.real(np.fft.ifft2(np.fft.ifftshift(out_dft[:, :, i])))

out[out < 0] = 0
out = out / out.max() * 255

out_dft_abs = np.abs(out_dft)
out_dft_abs = out_dft_abs / out_dft_abs.max() * 255
out_dft_abs = np.log(out_dft_abs + 1)
out_dft_abs = out_dft_abs / out_dft_abs.max() * 255
plt.imsave('res29-hybrid.jpg', out_dft_abs.astype(np.uint8))


far_out = cv2.resize(out.astype(np.uint8), (int(img_near.shape[1] / 10), int(img_near.shape[0] / 10)), interpolation=cv2.INTER_AREA)
plt.imsave("res30-hybrid-near.jpg", out.astype(np.uint8))
plt.imsave("res31-hybrid-far.jpg", far_out)


