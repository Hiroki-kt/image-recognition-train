import cv2
import numpy as np
import glob
import json
import os
import sys


def make_mask_img(img_url, view=False):
    gray_img = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)
    canny_img = cv2.Canny(gray_img, 200, 300)
    kernel = np.ones((500, 500), np.uint8)
    closing = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)
    if view:
        CANNY_WINDOW_NAME = "test"
        cv2.namedWindow(CANNY_WINDOW_NAME)
        cv2.imshow(CANNY_WINDOW_NAME, closing)
        # 終了処理
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return closing


def add_mask(img_url, mask_img, view=False):
    img = cv2.imread(img_url, cv2.IMREAD_COLOR)
    img[mask_img == 0] = [0, 0, 0]
    if view:
        CANNY_WINDOW_NAME = "test"
        cv2.namedWindow(CANNY_WINDOW_NAME)
        cv2.imshow(CANNY_WINDOW_NAME, img)
        # 終了処理
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


def get_imgs(data_path, output_path, mask_path):
    key_open = open('./data.json', 'r')
    key_json = json.load(key_open)
    for key_id in key_json:
        key = key_json[key_id]
        directory = os.listdir(output_path + '/labels')
        count = len(directory)
        for img_path in glob.glob(data_path + '/images/*' + key['img'] + '*.bmp'):
            # check the label
            label_path = img_path.replace('.bmp', '.txt')
            label_path = label_path.replace('images', 'labels')
            label = open(label_path)
            label_text = label.read()
            if label_text != "":
                print(img_path)
                # add mask to image and change to jpg
                mask_img = make_mask_img(
                    mask_path + '/mask-' + key['mask'] + '.bmp')
                masked_img = add_mask(img_path, mask_img)
                cv2.imwrite(output_path + '/images/' + str(count) +
                            '_' + key['mask'] + '.jpg', masked_img)
                # cp the label as new name
                new_label = open(output_path + '/labels/' + str(count) +
                                 '_' + key['mask'] + '.txt', 'w')
                new_label.write(label_text)
                label.close
                new_label.close
                count += 1


if __name__ == '__main__':
    args = sys.argv
    get_imgs(args[1], args[2], args[3])
