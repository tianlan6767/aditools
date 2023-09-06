import numpy as np


# def pad_im(im,pad_size,method="below"):
#   im_h,im_w = im.shape[:2]
#   pad_h,pad_w = pad_size
  
#   if im_w != pad_w and im_h != pad_h:
#       if method=="center":
#           _pad_w = round((pad_w-im_w)/2)
#           im = np.hstack(([np.zeros((im_h,_pad_w), np.uint8),im, np.zeros((im_h,pad_w-im_w-_pad_w), np.uint8)]))
#           _pad_h = round((pad_h-im_h)/2)
#           im = np.vstack(([np.zeros((_pad_h,pad_w), np.uint8),im, np.zeros((pad_h-im_h-_pad_h,pad_w), np.uint8)]))
#       elif method=="below":
#           im = np.hstack((im, np.zeros((im_h,pad_w-im_w), np.uint8)))
#           im = np.vstack((im, np.zeros((pad_h-im_h,pad_w), np.uint8)))
#   else:
#     if im_w != pad_w:
#       im = np.hstack((im, np.zeros((pad_h,pad_w-im_w), np.uint8)))
#     if im_h != pad_h:
#       im = np.vstack((im, np.zeros((pad_h-im_h,pad_w), np.uint8)))
#   return im


def pad_im(im,pad_size,method="below"):
    im_h,im_w = im.shape[:2]
    pad_h,pad_w = pad_size
    if len(im.shape)== 3:
        padding = ((0, pad_h-im_h), (0, pad_w-im_w), (0, 0))
    else:
        padding = ((0, pad_h-im_h), (0, pad_w-im_w))
    # if im_w != pad_w and im_h != pad_h:
        # im = np.hstack((im, np.zeros((im_h,pad_w-im_w), np.uint8)))
        # im = np.vstack((im, np.zeros((pad_h-im_h,pad_w), np.uint8)))
    return np.pad(im,padding,mode="constant",constant_values=0,)
    # else:
    #     if im_w != pad_w:
    #       im = np.hstack((im, np.zeros((pad_h,pad_w-im_w), np.uint8)))
    #     if im_h != pad_h:
    #       im = np.vstack((im, np.zeros((pad_h-im_h,pad_w), np.uint8)))
    