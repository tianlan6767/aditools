import cv2
import os
import json
import glob
import numpy as np
from abc import ABCMeta, abstractmethod


class BaseTransform(metaclass=ABCMeta):
    def _set_attributes(self, params):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def apply_im(self, im):
        return

    @abstractmethod
    def apply_polygon(self, polygon):
        """
          polygon:  N*2 [[(x,y)....(x,y)]
        """
        return

    def apply_polygons(self, polygons):
        if len(polygons) == 0:
            return []
        return [self.apply_polygon(np.asarray(p, dtype=int)) for p in polygons]


class BlendTransform:
    def __init__(self, im, src_image, src_weight, dst_weight):
        self.im = im
        self.src_image = src_image
        self.src_weight = src_weight
        self.dst_weight = dst_weight

    def __call__(self, *args, **kwargs):
        im = self.im.astype(np.float32)
        img = self.src_weight * self.src_image + self.dst_weight * im
        return np.clip(img, 0, 255).astype(np.uint8)


class TransformList(BaseTransform):
    def __init__(self, transforms):
        super().__init__()
        tfms = []
        for t in transforms:
            if isinstance(t, BaseTransform):
                tfms.append(t)
            else:
                raise ValueError(
                    "TransformList requires a list of Transform. Got type {type(t)}")
        self.transforms = tfms

    def __getitem__(self, item, id):
        return self.transforms[id]

    def __len__(self):
        return len(self.transforms)

    def __getattribute__(self, item):
        pass


class RotationTransform(BaseTransform):
    def __init__(self, angle, expand=True):
        super().__init__()
        # h, w = origin_shape[0], origin_shape[1]
        self.angle = angle
        self.expand = expand
        # self.center = np.array((w / 2, h / 2))
        # abs_cos, abs_sin = (abs(np.cos(np.deg2rad(self.angle))),
        #                     abs(np.sin(np.deg2rad(self.angle))))
        # if self.expand:
        #     self.new_w, self.new_h = np.rint(
        #         [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
        #     ).astype(int)
        # else:
        #     self.new_w, self.new_h = w, h

        self._set_attributes(locals())

        # Needed because of this problem https://github.com/opencv/opencv/issues/11784

    def apply_im(self, im):
        h, w = im.shape[:2]
        self.center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(self.angle))),
                            abs(np.sin(np.deg2rad(self.angle))))

        if self.expand:
            self.new_w, self.new_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            self.new_w, self.new_h = w, h
        self.dst_im = self.create_rotation_matrix(offset=-0.5)
        return cv2.warpAffine(im, self.dst_im, (self.new_w, self.new_h))

    def apply_polygon(self, polygon):
        self.dst_polygon = self.create_rotation_matrix()
        return cv2.transform(polygon[:, np.newaxis, :], self.dst_polygon)[:, 0, :]

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            rot_im_center = cv2.transform(
                self.center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array(
                [self.new_w / 2, self.new_h / 2]) + offset - rot_im_center
            rm[:, 2] += new_center
        return rm


class HFlipTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self._set_attributes(locals())

    def apply_im(self, im):
        self.h, self.w = im.shape[:2]
        return np.flip(im, axis=1)

    def apply_polygon(self, polygon):
        polygon[:, 0] = self.w - polygon[:, 0]
        return polygon


class VFlipTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self._set_attributes(locals())

    def apply_im(self, im):
        self.h, self.w = im.shape[:2]
        return np.flip(im, axis=0)

    def apply_polygon(self, polygon):
        polygon[:, 1] = self.h - polygon[:, 1]
        return polygon


class HVFlipTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self._set_attributes(locals())

    def apply_im(self, im):
        self.h, self.w = im.shape[:2]
        return np.flip(im, axis=None)

    def apply_polygon(self, polygon):
        polygon[:, 1] = self.h - polygon[:, 1]
        polygon[:, 0] = self.w - polygon[:, 0]
        return polygon


class ScaleTransform(BaseTransform):
    """
    target_size : (H*W)
    """

    def __init__(self, target_shape):
        super().__init__()
        assert len(target_shape) == 2, "transform size should be (H*W)"

        self._set_attributes(locals())

    def apply_im(self, im):
        self.h, self.w = im.shape[:2]
        if 0 < self.target_shape[0] < 1 and 0 < self.target_shape[0] < 1:
            self.new_h, self.new_w = int(
                self.target_shape[0]*self.h), int(self.target_shape[1]*self.w)
        else:
            self.new_h, self.new_w = int(
                self.target_shape[0]), int(self.target_shape[1])
        return cv2.resize(im, (self.new_w, self.new_h))

    def apply_polygon(self, polygon):
        polygon[:, 0] = polygon[:, 0] * (self.new_h * 1.0 / self.h)
        polygon[:, 1] = polygon[:, 1] * (self.new_w * 1.0 / self.w)
        return polygon


class ConstrastTransform(BaseTransform):
    def __init__(self, blend_args):
        super().__init__()
        self.blend_args = blend_args
        assert isinstance(
            blend_args, tuple), "ConstrastTransform input a tuple"
        assert blend_args[0] < blend_args[1], "min_value should be less than max_value"
        self._set_attributes(locals())

    def apply_im(self, im):
        value = np.random.uniform(self.blend_args[0], self.blend_args[1])
        return BlendTransform(im, src_image=im.mean(), src_weight=1 - value, dst_weight=value)()

    def apply_polygon(self, polygon):
        return polygon


class BrightnessTransform(BaseTransform):
    def __init__(self, blend_args):
        super().__init__()
        self.blend_args = blend_args
        assert self.blend_args[0] < self.blend_args[1], "min_value should be less than max_value"
        self._set_attributes(locals())

    def apply_im(self, im):
        value = np.random.uniform(self.blend_args[0], self.blend_args[1])
        return BlendTransform(im, src_image=0, src_weight=1 - value, dst_weight=value)()

    def apply_polygon(self, polygon):
        return polygon


# if __name__ == '__main__':
#     from copy import deepcopy
#     src = r"C:\Users\lubin\Desktop\3\del"
#     dst = r"C:\Users\lubin\Desktop\3\delout"
#     jf = r"C:\Users\lubin\Desktop\3\del\data_merge.json"
#     imgs = glob.glob(src + r'\*.[jb][pm][gp]')
#     json_data = json.load(open(jf))
#     new_json = deepcopy(json_data)
#     for img in imgs:
#         fn = os.path.basename(img)
#         new_json[fn] = {}
#         new_json[fn]["filename"] = fn
#         im = cv2.imread(img, 0)
#         h, w = im.shape[:2]
#         regions = json_data[fn]['regions']
#         polygons = [list(zip(i['shape_attributes']['all_points_x'], i['shape_attributes']['all_points_y'])) for i in
#                     regions]
#         print(regions)
#         r = RotationTransform((h, w), 5)

#         i = r.apply_im(im)
#         p = r.apply_polygons(polygons)
#         # po = np.array(p, np.int32).reshape((-1, 1, 2))
#         for index, region in enumerate(regions):
#             region['shape_attributes'].update({"all_points_x": p[index][:, 0]})
#             region['shape_attributes'].update({"all_points_y": p[index][:, 1]})
#         print(regions)


#         # cv2.polylines(img=i, pts=[po], isClosed=True, color=(0, 0, 255), thickness=4)
#         cv2.imwrite(os.path.join(dst, fn), i)
#         break
