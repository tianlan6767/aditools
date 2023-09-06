import os


class DppFileBase:
    def __init__(self):
        pass
      
    def isstr(self,param):
        """ 
        判断是否是文件名
        """
        return isinstance(param,str)
  
    def isfile(self,param):
        """ 
        判断是否是文件
        """
        return os.path.isfile(param)
      
    def ispath(self,param):
        """ 
        判断是否是路径
        """
        return os.path.isdir(param)
      
    def exists(self,path):
        """ 
        判断是否是存在路径
        """
        return os.path.exists(path)
      
    def create(self,path):
        """ 
        创建路径
        """
        if self.isfile(path) and not self.exists(path):
            os.makedirs(path)
            
    def filepath(self,param):
        """ 
        获取文件路径
        """
        if self.isfile(param):
            return os.path.dirname(param)
        else:
            return None
    
    def filename(self,param):
        """ 
        获取文件名
        """
        if self.isfile(param):
            return os.path.basename(param)
        elif self.isstr(param):
            return param
        else:
            return None
          
    def fmt(self,param):
        """ 
        获取后缀
        """
        return param.split(".")[-1]

    def recover_fn(self,param):
        """ 
        小图还原成原图名 1-2_2_2_0.jpg => 1-2_2_2.jpg
        """
        return "_".join(param.split("_")[:-1]) + "." + self.fmt(param)
      
    def getIndex(self,param):
        """ 
        小图还原成原图名 1-2_2_2_0.jpg => 0
        """
        return param.split("_")[-1].split('.')[0]
      
    def seg_filename(self,index,param):
        """
        重命名分割小图名 1-2_2_2.jpg =>1-2_2_2_s0.jpg
        """
        fmt = self.fmt(param)
        return "{}_s{}.{}".format(param.split(".")[0],index,fmt)
      
    def seg_dst(img_path,dst_name):
        """
        设定文件输出路径
        """
        return img_path +"/{}".format(dst_name)


DppFile = DppFileBase()