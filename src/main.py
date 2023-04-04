import cv2
import random

from clicker import select_coord_on_img
from distr import *


def open_img(root, num):
    pic = cv2.imread(root + str(num) + '.png', cv2.IMREAD_UNCHANGED)
    return pic


def show_img(img):
    plt.imshow(img, cmap="gray")
    plt.show()


def calc_mean(img, r, coords):
    sample = []
    back_col = 255

    rect_x = int(coords[0] - r / 2)
    rect_y = int(coords[1] - r / 2)

    for i in range(rect_x, rect_x + r):
        for j in range(rect_y, rect_y + r):
            if i < img.shape[0] and j < img.shape[1]:
                sample.append(img[i][j])
            else:
                sample.append(back_col)
    return np.mean(sample)


def get_data_from_user(img):
    show_img(img)

    r_1 = int(input("Enter first radius: "))
    x_1, y_1 = select_coord_on_img(img, r_1)

    r_2 = int(input("Enter second radius: "))
    x_2, y_2 = select_coord_on_img(img, r_2)

    u = (x_2 - x_1, y_2 - y_1)

    etalon_1 = calc_mean(img, r_1, (x_1, y_1))
    etalon_2 = calc_mean(img, r_2, (x_2, y_2))

    plt.imshow(img, cmap="gray")
    rect_1 = plt.Rectangle((x_1 - r_1 / 2, y_1 - r_1 / 2), width=r_1, height=r_1, fc='blue', alpha=0.4)
    rect_2 = plt.Rectangle((x_2 - r_2 / 2, y_2 - r_2 / 2), width=r_2, height=r_2, fc='blue', alpha=0.4)

    plt.gca().add_patch(rect_1)
    plt.gca().add_patch(rect_2)
    plt.show()

    return r_1, r_2, u, etalon_1, etalon_2


def measure_momental_profit(distr, real_value, new_predict, curr_predict=None):
    if curr_predict is None:
        curr_predict = distr.get_mean()

    curr_profit = 1 - distr.get_p_of_event(real_value, curr_predict)
    new_profit = 1 - distr.get_p_of_event(real_value, new_predict)
    momental_profit = new_profit - curr_profit

    return momental_profit


def get_one_pixel_sample(sample_size: int, img) -> list:
    full_sample = img.ravel()
    S_1 = [random.choice(full_sample) for _ in range(sample_size)]

    return S_1


def get_sample_for_bio_sensor(sensor_radius: int, sample_size: int, img) -> list:
    full_sample = img.ravel()
    r_squared = sensor_radius ** 2
    S_r = []

    for _ in range(sample_size):
        left = random.randint(0, len(full_sample) - r_squared)
        right = left + r_squared

        S_r.append(np.mean(full_sample[left:right]))

    return S_r


root = '../data/apple'
img_num = 100
img = open_img(root, img_num)
sample = img.ravel()
print(get_one_pixel_sample(10, img))
print(get_sample_for_bio_sensor(4, 10, img))

r_1, r_2, u, etalon_1, etalon_2 = get_data_from_user(img)

my_distr = Distr(0, 255, sorted(sample))
my_distr.draw()
print(measure_momental_profit(my_distr, 90, 105))
