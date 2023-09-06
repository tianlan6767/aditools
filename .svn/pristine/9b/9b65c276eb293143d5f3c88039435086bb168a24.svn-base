import math
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import os


def get_patches(A, a_r, a_c, B, b_r, b_c, patch_size=3):
    """
        获取A图与B图之间的匹配块

    :param A:
    :param a_r:
    :param a_c:
    :param B:
    :param b_r:
    :param b_c:
    :param patch_size: 匹配块尺寸，一般为3或5
    :return:
    """

    dx0 = dy0 = patch_size // 2
    dx1 = dy1 = patch_size // 2 + 1
    dx0 = min(a_c, b_c, dx0)
    dx1 = min(A.shape[1] - a_c, B.shape[1] - b_c, dx1)
    dy0 = min(a_r, b_r, dy0)
    dy1 = min(A.shape[0] - a_r, B.shape[0] - b_r, dy1)

    patch_A = A[a_r - dy0:a_r + dy1, a_c - dx0:a_c + dx1]
    patch_B = B[b_r - dy0:b_r + dy1, b_c - dx0:b_c + dx1]

    return patch_A, patch_B


def is_search(row, col, mask, search=0, patch_size=3) -> bool:
    """
        判断点(row, col)对应的patch块是否在搜索区域内

    :param row: patch中心点横坐标
    :param col: patch中心点纵坐标
    :param mask: 掩模信息
    :param search: 搜索区域的掩模值
    :param patch_size: 匹配块尺寸，一般为3或5
    :return:
    """

    patch = get_patches(mask, row, col, mask, row, col, patch_size)[0]
    return (patch == search).all()


def cal_distance(A, a_r, a_c, B, b_r, b_c, patch_size) -> float:
    """
        计算A图块与B图块之间的匹配误差，通过L2范数衡量误差大小

    :param A: A图
    :param a_r: A图块的中心点横坐标
    :param a_c: A图块的中心点纵坐标
    :param B: B图
    :param b_r: B图块的中心点横坐标
    :param b_c: B图块的中心点纵坐标
    :param patch_size: 块尺寸
    :return: 匹配误差
    """

    patch_A, patch_B = get_patches(A, a_r, a_c, B, b_r, b_c, patch_size)
    dist = np.linalg.norm(patch_A - patch_B)
    return dist


def init_NNF(A, mask=None, search=0, patch_size=3, match_ratio=0.5):
    """
        初始化最邻近场

    :param A: A图
    :param mask: 掩模图信息，定义了搜索区域和破损区域
    :param search: 搜索区域的掩模值
    :param patch_size: 匹配块尺寸，一般为3或5
    :param match_ratio: 限定匹配搜索框的边界, [0.3, 1]
    :return: (rows, cols, 2) Patch匹配偏移信息
    """

    rows, cols = A.shape[:2]
    if mask is not None:
        assert mask.shape == (rows, cols)
        assert 0.3 <= match_ratio <= 1

    def get_random_match_offset(_row, _col):
        """获取随机匹配信息"""
        _r = np.random.randint(0, rows)
        _r = _r if _r + _row < rows else _r - rows
        _c = np.random.randint(0, cols)
        _c = _c if _c + _col < cols else _c - cols
        return _r, _c

    nnf = np.zeros([rows, cols, 2], dtype=np.int32)
    for row in range(rows):
        for col in range(cols):
            offset_r, offset_c = get_random_match_offset(row, col)
            if mask is not None:
                # 搜索区域跳过
                if is_search(row, col, mask, search, patch_size):
                    continue
                # 确保随机匹配块在搜索区域，同时，限定匹配块不要偏移太远
                while not (is_search(row + offset_r, col + offset_c, mask, search, patch_size) and
                           abs(offset_r) < rows * match_ratio and abs(offset_c) < cols * match_ratio):
                    offset_r, offset_c = get_random_match_offset(row, col)
            nnf[row, col] = [offset_r, offset_c]
    return nnf


def init_NND(A, B, nnf, patch_size):
    """
        初始化A图与B图之间patch的匹配误差

    :param A: A图
    :param B: B图
    :param nnf: 最近邻场
    :param patch_size: 块尺寸
    :return: 匹配误差矩阵
    """

    nnd = np.zeros(A.shape[:2])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if nnf[i, j, 0] == nnf[i, j, 1] == 0:
                nnd[i, j] = 0
            else:
                nnd[i, j] = cal_distance(A, i, j, B, i + nnf[i, j, 0], j + nnf[i, j, 1], patch_size)
    return nnd


def propagation(A, a_r, a_c, B, nnf, nnd, patch_size, is_odd):
    """
        匹配传递

    :param A:
    :param a_r:
    :param a_c:
    :param B:
    :param nnf:
    :param nnd:
    :param patch_size:
    :param is_odd:
    :return:
    """
    best_dist = nnd[a_r, a_c]
    best_offset_r = nnf[a_r, a_c, 0]
    best_offset_c = nnf[a_r, a_c, 1]

    if best_dist == 0:
        return nnf, nnd

    if is_odd:
        # 计算(a_r + 1, a_c)的match块
        offset_r = nnf[a_r + 1, a_c, 0] - 1
        offset_c = nnf[a_r + 1, a_c, 1]
        if a_r + offset_r >= 0:
            dist = cal_distance(A, a_r, a_c, B, a_r + offset_r, a_c + offset_c, patch_size)
            if dist < best_dist:
                best_offset_r, best_offset_c, best_dist = offset_r, offset_c, dist

        # 计算(a_r, a_c + 1)的match块
        offset_r = nnf[a_r, a_c + 1, 0]
        offset_c = nnf[a_r, a_c + 1, 1] - 1
        if a_c + offset_c >= 0:
            dist = cal_distance(A, a_r, a_c, B, a_r + offset_r, a_c + offset_c, patch_size)
            if dist < best_dist:
                best_offset_r, best_offset_c, best_dist = offset_r, offset_c, dist
    else:
        # 计算(a_r - 1, a_c)的match块
        offset_r = nnf[a_r - 1, a_c, 0] + 1
        offset_c = nnf[a_r - 1, a_c, 1]
        if a_r + offset_r < B.shape[0]:
            dist = cal_distance(A, a_r, a_c, B, a_r + offset_r, a_c + offset_c, patch_size)
            if dist < best_dist:
                best_offset_r, best_offset_c, best_dist = offset_r, offset_c, dist

        # 计算(a_r, a_c - 1)的match块
        offset_r = nnf[a_r, a_c - 1, 0]
        offset_c = nnf[a_r, a_c - 1, 1] + 1
        if a_c + offset_c < B.shape[1]:
            dist = cal_distance(A, a_r, a_c, B, a_r + offset_r, a_c + offset_c, patch_size)
            if dist < best_dist:
                best_offset_r, best_offset_c, best_dist = offset_r, offset_c, dist

    nnf[a_r, a_c] = [best_offset_r, best_offset_c]
    nnd[a_r, a_c] = best_dist

    return nnf, nnd


def random_search(A, a_r, a_c, B, nnf, nnd, patch_size, attenuation_factor=0.5, mask=None, search=0, match_ratio=0.5):

    best_offset_r = nnf[a_r, a_c, 0]
    best_offset_c = nnf[a_r, a_c, 1]
    best_dist = nnd[a_r, a_c]

    if best_dist == 0:
        return nnf, nnd

    rows, cols = A.shape[:2]
    R_max = max(rows, cols)
    attenuation = 0
    while True:
        # 获取随机参数, 服从[-1, 1]均匀分布
        R_row = np.random.random() * 2 - 1.0
        R_col = np.random.random() * 1 - 1.0
        # 计算随机块的位置
        vec_row = R_max * pow(attenuation_factor, attenuation) * R_row
        vec_col = R_max * pow(attenuation_factor, attenuation) * R_col
        length = math.sqrt(vec_col * vec_col + vec_row * vec_row)
        if length < 1:
            break

        now_row = a_r + best_offset_r + int(vec_row)
        now_col = a_c + best_offset_c + int(vec_col)
        if 0 < now_col < cols - 1 and 0 < now_row < rows - 1 and abs(now_row - a_r) < match_ratio * rows and \
                (mask is None or is_search(now_row, now_col, mask, search, patch_size)):
            dist = cal_distance(A, a_r, a_c, B, now_row, now_col, patch_size)
            if dist < best_dist:
                best_dist = dist
                best_offset_r = now_row - a_r
                best_offset_c = now_col - a_c

        attenuation += 1

    nnf[a_r, a_c] = [best_offset_r, best_offset_c]
    nnd[a_r, a_c] = best_dist

    return nnf, nnd


def PatchMatch(A, B, patch_size, iterations, mask=None, search=0, match_ratio=0.5):
    """
        PatchMatch算法

    :param A:
    :param B:
    :param patch_size:
    :param iterations:
    :param mask:
    :param search:
    :param match_ratio: 限定匹配搜索框的边界, [0.3, 1]
    :return: 最近邻场矩阵
    """

    rows, cols = A.shape[:2]
    # 1.0 初始化近邻场
    nnf = init_NNF(A, mask=mask, patch_size=patch_size, search=search, match_ratio=match_ratio)
    # 1.1 计算近邻场的匹配误差
    nnd = init_NND(A, B, nnf, patch_size)
    pro_bar = tqdm(range(0, iterations), desc='PatchMatch', ncols=100)
    # 2.0 迭代
    for itr in pro_bar:
        is_odd = itr % 2
        if is_odd:
            # 奇次迭代, 从右至左, 从下至上
            # 奇次次, 对一个Patch查找其原对应点右(x + 1, y), 下(x, y + 1)的Patch
            for i in range(rows - 2, -1, -1):
                for j in range(cols - 2, -1, -1):
                    propagation(A, i, j, B, nnf, nnd, patch_size, is_odd)
                    random_search(A, i, j, B, nnf, nnd, patch_size, mask=mask, match_ratio=match_ratio, search=search)
        else:
            # 偶次迭代, 从左至右, 从上至下
            # 偶次次, 对一个Patch查找其原对应点左(x - 1, y), 上(x, y - 1)的Patch
            for i in range(1, rows):
                for j in range(1, cols):
                    # 2.1 匹配传递
                    propagation(A, i, j, B, nnf, nnd, patch_size, is_odd)
                    # 2.2 随机搜索
                    random_search(A, i, j, B, nnf, nnd, patch_size, mask=mask, match_ratio=match_ratio, search=search)

    return nnf


def build_image(nnf, B):
    """
        根据最近领场(NNF),重建图像

    :param nnf: 最近领场
    :param B: 目标图像
    :return:
    """

    rows, cols = nnf.shape[:2]
    out_shape = [rows, cols, B.shape[2]] if B.ndim == 3 else [rows, cols]
    out_image = np.zeros(out_shape, dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            pick_row = nnf[i, j][0] + i
            pick_col = nnf[i, j][1] + j
            if B.ndim == 3:
                out_image[i, j, :] = B[pick_row, pick_col, :]
            else:
                out_image[i, j] = B[pick_row, pick_col]
    return out_image


def method1():
    from comm import get_cur_time
    A = np.array(Image.open(r"D:\Projects\PS\PatchMath\images\defect\src20.bmp"))
    B = np.array(Image.open(r"D:\Projects\PS\PatchMath\images\defect\src20.bmp"))
    mask = np.array(Image.open(r"D:\Projects\PS\PatchMath\images\defect\mask20.bmp"))
    mask = None

    # n_mask = cv2.bitwise_not(mask)
    # gray_mean = cv2.mean(A, n_mask)[0]
    # o_mask = (mask / 255 * int(gray_mean)).astype(np.uint8)
    # A = cv2.bitwise_and(A, n_mask)
    # A = cv2.add(A, o_mask)

    nnf = PatchMatch(A, B, patch_size=3, iterations=4, mask=mask, match_ratio=1, search=0)
    dst = build_image(nnf, A)

    Image.fromarray(np.uint8(dst)).show()
    img_name = f'{get_cur_time()}.bmp'
    # Image.fromarray(np.uint8(dst)).save(os.path.join(r"D:\Projects\PS", img_name))


def method2():
    img = np.array(Image.open(r"D:\Projects\PS\src.bmp"))
    mask = np.array(Image.open(r"D:\Projects\PS\mask2.bmp"))
    # n_mask = cv2.bitwise_not(mask)
    # n_mask = cv2.cvtColor(n_mask, cv2.COLOR_GRAY2BGR)
    # img = cv2.bitwise_and(img, n_mask)

    pyramid_num = 3
    pyr_nnf = None
    # for pyramid_level in range(pyramid_num, -1, -1):
    #     factor = 1.0 / pow(2, pyramid_level)
    #     pyr_img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
    #     pyr_msk = cv2.resize(mask, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
    #     _, pyr_msk = cv2.threshold(pyr_msk, 0, 255, cv2.THRESH_BINARY)
    #     # Image.fromarray(np.uint8(pyr_img)).show()
    #     # Image.fromarray(np.uint8(pyr_msk)).show()
    #
    #     if pyramid_level == pyramid_num:
    #         pyr_nnf = init_NNF(pyr_img, pyr_msk)
    #     else:
    #         rows, cols = pyr_nnf.shape[:2]
    #         nnf_rows, nnf_cols = pyr_img.shape[:2]
    #         new_pyr_nnf = np.zeros([nnf_rows, nnf_cols, 2], dtype=np.int32)
    #         for r in range(rows):
    #             r2 = r * 2
    #             for c in range(cols):
    #                 c2 = c * 2
    #                 if r2 < nnf_rows and c2 < nnf_cols:
    #                     new_pyr_nnf[r2, c2] = pyr_nnf[r, c] * 2
    #                 if r2 + 1 < nnf_rows and c2 < nnf_cols:
    #                     new_pyr_nnf[r2 + 1, c2] = pyr_nnf[r, c] * 2
    #                 if r2 < nnf_rows and c2 + 1 < nnf_cols:
    #                     new_pyr_nnf[r2, c2 + 1] = pyr_nnf[r, c] * 2
    #                 if r2 + 1 < nnf_rows and c2 + 1 < nnf_cols:
    #                     new_pyr_nnf[r2 + 1, c2 + 1] = pyr_nnf[r, c] * 2
    #         pyr_nnf = new_pyr_nnf
    #     pyr_nnf = PatchMatch(pyr_img, pyr_img, patch_size=3, iterations=4, mask=pyr_msk)
    #
    # dst = build_image(pyr_nnf, img)
    # Image.fromarray(np.uint8(dst)).show()


def method3():
    img = np.array(Image.open(r"D:\Projects\PS\src.bmp"))
    mask = np.array(Image.open(r"D:\Projects\PS\mask2.bmp"))

    dst = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    Image.fromarray(np.uint8(dst)).show()


if __name__ == '__main__':
    # img = np.array(Image.open(r"D:\Projects\PS\mask5.bmp"))
    # mask = np.array(Image.open(r"D:\Projects\PS\mask51.bmp"))
    # img = cv2.add(img, mask)
    # Image.fromarray(np.uint8(img)).save(os.path.join(r"D:\Projects\PS", 'mask52.bmp'))

    method1()
