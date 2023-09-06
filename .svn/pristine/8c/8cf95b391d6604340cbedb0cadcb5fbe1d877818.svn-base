def XYXY_XY0XY(polygons,step=2):
    """
    polygons: [x,y,x,y....x,y]
    returns: [(x,y),(x,y).....(x,y)]
    """
    assert isinstance(polygons,list),"polygons must be type of list"
    return [(polygons[i],polygons[i+1]) for i in range(0, len(polygons), step)]
    

def XX0YY_XY0XY(polygons):
    """
    polygons: [[x,x,....x],[y,y,....y]]
    returns: [(x,y),(x,y).....(x,y)]
    """
    assert len(polygons)==2,"polygons must be [[x,x,....x],[y,y,....y]] format"
    return [(x,y) for x,y in zip(polygons[0],polygons[1])]


####################GT_BOX#######################################################
def XX0YY_XYXY(polygons):
    """
    polygons: [[x,x,....x],[y,y,....y]]
    returns: [x_min,ymin,x_max,y_max]
    """
    assert len(polygons)==2,"polygons must be [[x,x,....x],[y,y,....y]] format"
    return [min(polygons[0]),min(polygons[1]),max(polygons[0]),max(polygons[1])]


def XX0YY_XYWH(polygons):
    """
    polygons: [[x,x,....x],[y,y,....y]]
    returns: [x_min,ymin,W,H]
    """
    assert len(polygons)==2,"polygons must be [[x,x,....x],[y,y,....y]] format"
    return [min(polygons[0]),min(polygons[1]),max(polygons[0])-min(polygons[0]),max(polygons[1])-min(polygons[1])]

# if __name__ == "__main__":
#     a = XX0YY_XYWH([[22.5, 123.5, 24.5, 125.5, 26.5, 127.5, 28.5, 129.5, 28.5, 126.5],[22.5, 123.5, 24.5, 125.5, 26.5, 127.5, 28.5, 129.5, 28.5, 126.5]])
#     print(a)
