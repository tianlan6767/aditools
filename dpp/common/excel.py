import numpy as np
from io import BytesIO
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Font,Border, Side, colors
from PIL import Image as IM


wb=Workbook()
sheet=wb.active


class EImage(Image):
    _id = 1
    _path = "/xl/media/image/{0}.{1}"
    anchor = "A1"

    def __init__(self, img):
        super(EImage, self).__init__(img)
        self.img = img
        self.width, self.height = self.img.size
        try:
            self.format = self.img.format.lower()
        except AttributeError:
            self.format = "jpeg"

    def _data(self):
        fp = BytesIO()
        self.img.save(fp, format="jpeg")
        fp.seek(0)
        return fp.read()

    @property
    def path(self):
        return self._path.format(self._id, self.format)


class Field:
    pass
        

class CharField(Field):
    def __init__(self, column_name, title_font=u'微软雅黑', title_size=16, title_height=25,
                 body_font=u'微软雅黑', body_size=12, body_height=44,column_width=8):
        self._value = None
        self.column_name = column_name
        self.title_font = title_font
        self.title_size = title_size
        self.title_height = title_height
        self.body_font = body_font
        self.body_size = body_size
        self.body_height = body_height
        self.column_width = column_width

    def __set__(self, instance, value):
        if isinstance(value, np.ndarray):
            raise ValueError("value must be str,int or float")
        else:
            self._value = value

    def __get__(self, instance, owner):
        return self._value


class ImageField(Field):
    def __init__(self, column_name, title_font=u'微软雅黑', title_size=16, title_height=25,column_width=8):
        self._value = None
        self.column_name = column_name
        self.title_font = title_font
        self.title_size = title_size
        self.title_height = title_height
        self.column_width = column_width

    def __set__(self, instance, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("Image value must be np.ndarray")
        else:
            self._value = value

    def __get__(self, instance, owner):
        return self._value


class ModelMetaClass(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        if name == 'BaseExcel':
            return super().__new__(cls, name, bases, attrs, **kwargs)
        mapping = {}
        for k, v in attrs.items():
            if isinstance(v, Field):
                mapping[k] = v
        attrs["mapping"] = mapping
        attrs["Excel"] = name
        for k in mapping.keys():
            attrs.pop(k)
        return super().__new__(cls, name, bases, attrs, **kwargs)


class BaseExcel(metaclass=ModelMetaClass):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        super(BaseExcel, self).__init__(**kwargs)

    def save(self, row):
        column = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                'U', 'V','W', 'X', 'Y', 'Z','AA', 'AB','AC', 'AD','AE', 'AF','AG', 'AH','AI', 'AJ','AK', 'AL','AM', 
                'AN','AO','AP','AQ', 'AR','AS', 'AT','AU', 'AV','AW', 'AX','AY', 'AZ']
        index = 1
        for key, value in self.mapping.items():
            item = getattr(self, key)
            if row == 0:
                font = Font(name=getattr(value, 'title_font'), size=getattr(value, 'title_size'))
                height = getattr(value, 'title_height')
                column_width = getattr(value, 'column_width')
                sheet["{}{}".format(column[index], str(row + 1))] = value.column_name
                sheet["{}{}".format(column[index], str(row + 1))].font = font
                sheet.row_dimensions[int(row + 1)].height = height
                sheet.column_dimensions[column[index]].width = column_width
            if isinstance(value, ImageField):
                im = EImage(IM.fromarray(item))
                im.anchor = "{}{}".format(column[index], str(row + 2))
                sheet.row_dimensions[int(row + 2)].height = 44
                im.width = im.width * 0.5
                im.height = im.height * 0.5
                sheet.add_image(im)
            else:
                font = Font(name=getattr(value, 'body_font'), size=getattr(value, 'body_size'))
                sheet["{}{}".format(column[index], str(row + 2))] = item
                sheet["{}{}".format(column[index], str(row + 2))].font = font
            index += 1

