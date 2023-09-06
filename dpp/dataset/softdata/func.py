import os,glob,json,cv2,math
from dpp.common.util import *
from PIL import Image,ImageFont,ImageDraw
from tqdm import tqdm


def get_img_dict(img_path,jf,crop_size,limit_w):
    all_imgs = []
    json_data = json.load(open(jf))
    for folder in tqdm(os.listdir(img_path),desc="生成缺陷小图参数字典。。。"):
        root = os.path.join(img_path,folder)
        if os.path.isdir(root):       # 只操作文件夹
            imgs = glob.glob(root+'\*.[jb][pm][gp]')
            folder_imgs = []
            for index_img,img in enumerate(imgs):
                filename = os.path.basename(img)
                im = cv2.imdecode(np.fromfile(img, dtype=np.uint8), 1)
                im_ori = im.copy()
                regions = json_data[filename]["regions"]
                for index,region in enumerate(regions):
                    xs,ys,label = parse_region(region)
                    area = cal_area(xs,ys)
                    cv2.polylines(img=im, pts=[np.dstack((xs, ys))], isClosed=True,color=(0,255,0), thickness=1)
                    start_x,end_x,start_y,end_y = scale_small_img(im.shape[:2],(xs,ys),crop_size=crop_size)
                    fn = filename.split('.')[0] + '_{}'.format(index) + '.jpg'
                    img_mask_region = im[start_y:end_y, start_x:end_x]
                    img_ori_region = im_ori[start_y:end_y, start_x:end_x]
                    # cv2.imwrite(os.path.join(os.path.dirname(jf),fn),np.array(img_mask_region))
                    stack_im = np.hstack((img_ori_region,img_mask_region))
                    h,w = stack_im.shape[:2]
                    if h>limit_w and w>limit_w and h>w:
                        stack_im = cv2.resize(stack_im,(limit_w,int(limit_w/h*w)))
                        h,w = stack_im.shape[:2]
                    elif h>limit_w and w>limit_w and w>h:
                        stack_im = cv2.resize(stack_im,(int(limit_w/w*h),limit_w))
                        h,w = stack_im.shape[:2]
                    elif h>limit_w:
                        stack_im = cv2.resize(stack_im,(limit_w,int(limit_w/h*w)))
                        h,w = stack_im.shape[:2]
                    elif w>limit_w:
                        stack_im = cv2.resize(stack_im,(int(limit_w/w*h),limit_w))
                        h,w = stack_im.shape[:2]
                    cv2.putText(stack_im, filename.replace(".jpg",""), (5,15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    cv2.putText(stack_im, "A:"+str(area), (5,30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    pad_im = np.ones((h,5,3),dtype=np.uint8)*255
                    new_im = np.hstack((stack_im,pad_im))
                    h,w = new_im.shape[:2]
                    folder_imgs.append({"filename":filename,"fn":fn,"area":area,"path":root,"im":new_im,"index_img":index_img,"h":h,"w":w})
                # break
            all_imgs.append(folder_imgs)
    return all_imgs
  

def split_list(imgs_list ,max_w):
    """
    对一列表中前n位大于20时分段 [3, 6, 7, 8,4,5,6,1,5,6,7] => [[3, 6, 7], [8, 4, 5], [6, 1, 5, 6]]
    """
    s = 0 
    ws =[img["w"] for img in imgs_list]
    all_list = []
    per_list = []
    if sum(ws)<max_w:
        return imgs_list,0
    else:
        for item in ws:
            s += item
            if s<max_w:
                per_list.append(item)
            else:
                all_list.append(per_list)
                per_list =[]
                s = 0
                s+=item
                per_list.append(item)

        if len(all_list) == 1:
            return imgs_list,0
        else:
            new_imgs_list = []
            start = 0
            for imgs in all_list:
                length = len(imgs)
                new_imgs_list.append(imgs_list[start:start+length])
                start += length
            return new_imgs_list,1

  
def sorted_imgs(imgs_list):
    new_imgs_list = []
    for imgs in imgs_list:
        imgs = sorted(imgs, key=lambda x: x["area"],reverse=True)        
        new_imgs_list.append(imgs)
        
        
def gen_long_img(all_imgs,max_w):
    """
    遍历每个文件夹，各自生成一张长图，包含标题 max_h*max_w
    """
    full_im = []
    for imgs in all_imgs:
        max_h = max([img["h"] for img in imgs])
        start_w = 0
        paste_im = Image.fromarray(np.ones((max_h+30,int(max_w*1.6),3),dtype=np.uint8)*255)
        title_im = Image.fromarray(np.ones((30,max_w, 3),dtype=np.uint8)*255)
        fontpath = "dpp\dataset\softdata\SimSun.ttf"
        # 创建字体对象，并且指定字体大小
        font = ImageFont.truetype(fontpath, 18)
        # 把array格式转换成PIL的image格式
        # 创建一个可用来对其进行draw的对象
        draw = ImageDraw.Draw(title_im)
        if "fn" in imgs[0]:
            draw.text((2, 2), imgs[0]["path"], font=font, fill=(0,0,255))
        # draw.text((2, 2), os.path.basename(imgs[0]["path"]), font=font, fill=(0,0,255))
        paste_im.paste(title_im,(0,0,max_w,30))
        
        for img in imgs:
            h,w = img["h"],img["w"]
            paste_im.paste(Image.fromarray(img["im"]),(start_w,30,start_w+w,h+30))
            start_w+=w
        full_im.append(np.array(paste_im))
    result = np.concatenate(full_im,axis=0)
    return result
    

def gen_full_img(img_path,jf,crop_size,limit_w,max_w):
    all_imgs = get_img_dict(img_path,jf,crop_size,limit_w)
    new_all_imgs = []
    for item in all_imgs:
        imgs,status = split_list(item,max_w)
        if status==1:
            for i in imgs:
                new_all_imgs.append(i)
        else:
            new_all_imgs.append(imgs)
    long_im = gen_long_img(new_all_imgs,max_w)
    h,w = long_im.shape[:2]
    pad_im = np.ones((h,10,3),dtype=np.uint8)*255
    new_im = np.hstack((pad_im,long_im))
    cv2.imencode(".jpg",new_im)[1].tofile(os.path.join(img_path,"{}.jpg".format(os.path.basename(img_path))))
    
    
def split_list_average_n(origin_list, n):
    for i in range(0, len(origin_list), n):
        yield origin_list[i:i + n]
        

def get_no_json_long_img(img_path,max_w,max_size):
    all_imgs = []
    for root,dirs,files in os.walk(img_path):
        # if files and os.path.basename(root)!= "0" and os.path.basename(root)!= "1":
        if files: #  and os.path.basename(os.path.dirname(root))=="data"
            imgs = load_file(root, format="img")
            if len(imgs):
                print(root)
                folder_imgs = []
                for img in imgs:
                    im = cv2.imdecode(np.fromfile(img, dtype=np.uint8), 1)
                    h,w = im.shape[:2]
                    
                    if w>max_size*2 and h>max_size and h>w/2:
                        im = cv2.resize(im,(max_size,int(max_size/h*w)))
                    elif w>max_size and h>max_size and h<w/2:
                        im = cv2.resize(im,(int(max_size/w*h),max_size))
                    elif w>max_size*2 :
                        im = cv2.resize(im,(int(max_size/w*h),max_size))
                    elif h>max_size :
                        im = cv2.resize(im,(max_size,int(max_size/h*w)))
                    h,w = im.shape[:2]
                    pad_im = np.ones((h,8,3),dtype=np.uint8)*255
                    new_im = np.hstack((im,pad_im))
                    h,w = new_im.shape[:2]
                    folder_imgs.append({"filename":img,"fn":img,"area":"area","path":root,"im":new_im,"index_img":"1","h":h,"w":w})

                ws = sum([item["w"] for item in folder_imgs])
                if ws>max_w*1.01:
                    length = math.ceil(ws/max_w)
                    items = split_list_average_n(folder_imgs, math.ceil(len(folder_imgs)/length))
                    for index,item in enumerate(items):
                        if index>0:
                            [i.pop("fn") for i in item]
                        all_imgs.append(item)
                        
                else:             
                    all_imgs.append(folder_imgs)
                    
    return all_imgs