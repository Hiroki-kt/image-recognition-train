from PIL import Image, ImageDraw


def show_bb(img, x, y, w, h, text, textcolor, bbcolor):
    draw = ImageDraw.Draw(img)
    text_w, text_h = draw.textsize(text)
    label_y = y if y <= text_h else y - text_h
    draw.rectangle((x, label_y, x+w, label_y+h), outline=bbcolor)
    draw.rectangle((x, label_y, x+text_w, label_y+text_h),
                   outline=bbcolor, fill=bbcolor)
    draw.text((x, label_y), text, fill=textcolor)


if __name__ == '__main__':
    # img = Image.open('./output/voc-sep-output/JPEGImages/2_o-left_3.jpg')
    img = Image.open('./output/voc-output/JPEGImages/2_o-left.jpg')
    show_bb(img, 1392, 1674, 1016, 121, "wrinkles",
            (255, 255, 255), (255, 0, 0))
    show_bb(img, 1495, 1733, 31, 34, "blot", (255, 255, 255), (255, 0, 0))
    img.show()
