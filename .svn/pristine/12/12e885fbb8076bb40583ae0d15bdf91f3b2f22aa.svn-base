import cv2
import os
from tqdm import tqdm
from enum import Enum, unique


@unique
class FormatEnum(Enum):
    J2B = {"before": ".jpg", "after": ".bmp"}
    B2J = {"before": ".bmp", "after": ".jpg"}
    J2J = {"before": ".jpg", "after": ".jpg"}
    B2B = {"before": ".bmp", "after": ".bmp"}
    

def convert_format(imgs, dst, format, out_channel):
    format_value = FormatEnum[format].value
    before, after = format_value["before"], format_value["after"]
    for img in tqdm(imgs):
        filename = os.path.basename(img)
        im = cv2.imread(img, -1)
        if out_channel == 3 and len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if out_channel == 1 and len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(dst, filename.replace(before, after)), im)



