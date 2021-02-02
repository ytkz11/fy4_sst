# -*- coding: utf-8 -*-
# @Time    : 2021/1/29 11:39
# @Author  : dky
# @Software: PyCharm
import h5py
import netCDF4
import numpy as np
from matplotlib import pyplot as plt
import os, re, fnmatch, glob
import matplotlib
plt.rcParams['font.sans-serif']=['Kaiti']

def Read_Data(fy4file, hy1cfile, fy4_hy1c_file):
    # 读取sst要素
    in_Fy4_File = netCDF4.Dataset(fy4file, mode='r')
    # 这里是FY4A的变量，目前是固定的（如果后续发生命名变化，这里需要改）
    fy4a_var = 'Data'

    fy4a_SST = in_Fy4_File[fy4a_var][:].astype(np.float).copy()
    # 设置FY4A南海区域经纬度的大小
    fy4a_Lon_Size = 501
    fy4a_Lat_Size = 526
    # 设置无效值
    fy4a_SST[fy4a_SST == 65530] = np.NAN
    fy4a_SST[fy4a_SST == -888] = np.NAN

    # 自定义FY4A的南海范围经纬度，间隔为0.04°
    lon = np.linspace(103, 123, fy4a_Lon_Size)
    lat = np.linspace(25, 4, fy4a_Lat_Size)

    # 网格化
    [fy4a_grid_lat1, fy4a_grid_lon1] = np.meshgrid(lat, lon)
    fy4a_grid_lon2 = fy4a_grid_lon1.T
    fy4a_grid_lat2 = fy4a_grid_lat1.T
    #  把SST导入numpy
    original_SST = np.array(fy4a_SST)
    #  SST_2是南海区域，风云四号的海温数据
    original_Fy4a_SST = original_SST[725:1251, 825:1326]
    original_Fy4a_SST_mean = np.nanmean(original_Fy4a_SST, axis=1)

    '''
    HY1C
    '''
    # 读取sst要素
    in_file = h5py.File(hy1cfile, mode='r')
    # 这里是HY1C的变量，目前是固定的（如果后续发生命名变化，这里需要改）
    var = '/Geophysical Data/SST'
    hy1c_SST = in_file[var][:].astype(np.float).copy()
    hy1c_SST[hy1c_SST == -999] = np.NAN

    # 设置FY4A南海区域经纬度的大小
    hy1c_Lon_Size = 540
    hy1c_Lat_Size = 567
    # 自定义海洋一号的南海经纬度
    hy1c_lon = np.linspace(103.02264, 122.9871, hy1c_Lon_Size)
    hy1c_lat = np.linspace(24.9948, 4.03016, hy1c_Lat_Size)

    [hy1c_lat1, hy1c_lon1] = np.meshgrid(hy1c_lat, hy1c_lon)  # 网格化
    hy1c_lat2 = hy1c_lat1.T  # 转置
    hy1c_lon2 = hy1c_lon1.T  # 转置
    # 截取南海的经纬度
    hy1c_SST1 = hy1c_SST[1755:2322, 7641:8181]
    original_Hy1c_SST_mean = np.nanmean(hy1c_SST1, axis=1)

    '''
    FT4_HY1C
    '''
    lon3 = np.linspace(103, 123, 1001)  # 1001
    lat3 = np.linspace(25, 4, 1051)  # 1051
    in_Fy4_Hy1c_File = netCDF4.Dataset(fy4_hy1c_file, mode='r')
    fy4a_var = 'SST'

    fy4a_Hy1c_SST = in_Fy4_Hy1c_File[fy4a_var][:].astype(np.float).copy()
    original_fy4a_Hy1c_SST_mean = np.nanmean(fy4a_Hy1c_SST, axis=1)
    return original_Fy4a_SST_mean, original_Hy1c_SST_mean, original_fy4a_Hy1c_SST_mean, lat, hy1c_lat, lat3


def temperature_imshow(fy4file, hy1cfile, fy4_hy1c_file):

    # png的输出目录
    savepath = os.path.join(filepath, 'png')
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    #调用Read_Data函数
    y1, y2, y3, x1, x2, x3 = Read_Data(fy4file, hy1cfile, fy4_hy1c_file)
    step = 1
    y1 = y1[0:len(y1):step]
    y2 = y2[0:len(y2):step]
    y3 = y3[0:len(y3):step]
    x1 = x1[0:len(x1):step]
    x2 = x2[0:len(x2):step]
    x3 = x3[0:len(x3):step]
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'fontproperties': 'Kaiti',
            'size': 16,
            }
            
    l1 = plt.plot(x1, y1, '-', c='r', label='风云四号海温', lw=3)
    l2 = plt.plot(x2, y2, '-', c='g', label='海洋一号海温', lw=3)
    l3 = plt.plot(x3, y3, '-', c='b', label='融合海温', lw=3)
    
    plt.plot(x1, y1, 'r', x2, y2, 'g', x3, y3, 'b')

    name = os.path.basename(fy4_hy1c_file)
    time1 = re.findall(r"\d+.?\d", name)
    time2 = "".join(time1)
    if len(time2) == 8:
        year = time2[0:4]
        month = time2[4:6]
        day = time2[6:8]
        plt.title('自我数据对比 ' + month + '-' + day + ',' + year)
        plt.xlabel('纬度')
        plt.ylabel('温度')
        plt.legend()

        savefile = os.path.join(savepath, time2 + ".jpg")
        plt.savefig(savefile, dpi=1000)
        print('The image save in: ', savefile)
    else:
        plt.title('A comparison of three types of data')
        plt.xlabel('Latitude')
        plt.ylabel('Temperature')
        plt.legend()

        savefile = os.path.join(savepath, name+ ".jpg")
        plt.savefig(savefile, dpi=1000)
        print('The image save in: ', savefile)
    del y1, y2, y3, x1, x2, x3, l1, l2, l3
    plt.cla()
def match_file1():
    fy4filename=[]
    hy1cfilename = []
    fy4f_hy1c_filename = []
    for filename1 in os.listdir(fy4path):
        if fnmatch.fnmatch(filename1, '*.HDF'):  # 匹配模式为星号，表示任意的字符
            fy4filename.append(filename1)
            # print(filename1)
    for filename2 in os.listdir(hy1cpath):
        if fnmatch.fnmatch(filename2, '*.h5'):  # 匹配模式为星号，表示任意的字符
            hy1cfilename.append(filename2)
            # print(filename2)
    for filename3 in os.listdir(fy4_hy1c_path):
        if fnmatch.fnmatch(filename3, '*.nc'):  # 匹配模式为星号，表示任意的字符
            fy4f_hy1c_filename.append(filename3)
            # print(filename3)
    return fy4filename, hy1cfilename, fy4f_hy1c_filename

def Find_file(    fy4path , hy1cpath, fy4_hy1c_path):
    fy4filename, hy1cfilename, fy4f_hy1c_filename = match_file1()
    i = 0
    for file in fy4filename:
        date = file[18:26]
        file1 = os.path.join(fy4file,file)
        os.chdir(hy1cpath)
        file2 = glob.glob('*'+date+'*')
        file
        os.chdir(fy4_hy1c_path)
        file3 = glob.glob('*'+date+'*')

        temperature_imshow(file1, file2, file3)
        i=i+1
        print(i)


def run(fy4path , hy1cpath, fy4_hy1c_path):
    fy4filename, hy1cfilename, fy4f_hy1c_filename = match_file1()

    for file in fy4filename:
        date = file[18:26]

        print('match1',file, ':',date)

        file2 = fnmatch.filter(hy1cfilename, '*'+date+'*')
        file3 = fnmatch.filter(fy4f_hy1c_filename, '*'+date+'*')
        file2 = "".join(file2)
        file3 = "".join(file3)
        file1 = os.path.join(fy4path,file)
        file2 = os.path.join(hy1cpath, file2)
        file3 = os.path.join(fy4_hy1c_path, file3)
        if file1 !=[]:
            if file2 != []:
                if file3 != []:
                    print(file1, '/n', file2, '/n', file3)
                    temperature_imshow(file1, file2, file3)






if __name__ == '__main__':
    fy4path = r'D:\github\FY4_data\fy4'
    hy1cpath = r'D:\github\FY4_data\hy1c'
    fy4_hy1c_path = r'D:\github\FY4_data\out'

    filepath = r'D:\github\FY4_data\compare'


    run(fy4path , hy1cpath, fy4_hy1c_path)
    Find_file()

    b = 1
