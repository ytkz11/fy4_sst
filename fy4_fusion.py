# -*- coding: utf-8 -*-
# @Time    : 2021/1/29 10:35
# @Author  : dky
# @Software: PyCharm
import os
import sys
import time
import h5py
import netCDF4
import numpy as np
from pylab import mpl
import scipy.io as sio
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
import matplotlib
import re

print(matplotlib.matplotlib_fname())


def get_date(file):
    a = 1


def Interpolate_To_002_Degrees(original_Longitude, original_Latitude, data1, x, y):
    # 插值为0.02°间隔
    # data4是风云四号、海洋一号融合数据，三列数据，依次是经度、纬度、海温
    # 自定义0.02°间隔的南海经纬度

    lon3 = np.linspace(103, 123, x)  # 1001
    lat3 = np.linspace(25, 4, y)  # 1051
    [lat3, lon3] = np.meshgrid(lat3, lon3)  # 网格化
    lat4 = lat3.T  # 转置
    lon4 = lon3.T  # 转置

    data2 = Arranged_In_Three_Columns(original_Longitude, original_Latitude, data1)
    output_data1 = griddata(data2[:, 0:2], data2[:, 2], (lon4, lat4), method='linear')
    return output_data1, lon4, lat4


def Kriging(lon1, lat1, SST, lon_size, lat_size):
    lon, lat, z = Arranged_In_Three_Ros(lon1, lat1, SST)
    # 读取z的长度,此时的z为1*n的矩阵，读取z的列数即可
    z_Col = np.size(z, 1)

    # 建立克里金模型需要的参数，经度纬度温度的间隔步长为100
    lon = lon[:, 1:z_Col:100]
    lat = lat[:, 1:z_Col:100]
    z = z[:, 1:z_Col:100]

    # 建立传统克里金模型
    OK = OrdinaryKriging(
        lon,
        lat,
        z,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic", )

    # 生成526 x 501 的经纬度网格:
    # 生成567 x 540 的经纬度网格:

    # grid_lon = np.linspace(103, 123, 526)
    # grid_lat = np.linspace(25, 4, 501)

    grid_lon = np.linspace(103, 123, lon_size)
    grid_lat = np.linspace(25, 4, lat_size)

    # z2是插值后的数据，ss2是Sigma²:
    z2, ss2 = OK.execute("grid", grid_lon, grid_lat)
    # 把克里金插值得到的数据，转成为lon_size x lat_size大小的矩阵
    # z3 = np.array(z2)
    # z3 = z3.T

    # 构建FY4A经纬度网格
    [grid_lon1, grid_lat1] = np.meshgrid(grid_lon, grid_lat)
    # 转置
    grid_lon1 = grid_lon1.T
    grid_lat1 = grid_lat1.T
    return z2


# 输出三行数据，依次是经度、纬度、海温
def Arranged_In_Three_Ros(data1, data2, data3):
    # 降维
    d1 = data1.flatten()
    d2 = data2.flatten()
    d3 = data3.flatten()

    # 转为列的形式
    dd1 = d1[:, np.newaxis]
    dd2 = d2[:, np.newaxis]
    dd3 = d3[:, np.newaxis]

    data4 = np.c_[dd1, dd2, dd3]

    # 除去存在nan的行，过滤数据
    data5 = np.delete(data4, np.where(np.isnan(data4))[0], axis=0)
    ddd1 = data5[:, 0].flatten()
    ddd2 = data5[:, 1].flatten()
    ddd3 = data5[:, 2].flatten()

    # 转成行的形式
    ddd1 = ddd1[np.newaxis, :]
    ddd2 = ddd2[np.newaxis, :]
    ddd3 = ddd3[np.newaxis, :]
    return ddd1, ddd2, ddd3


# 输出三列数据，依次是经度、纬度、海温
def Arranged_In_Three_Columns(data1, data2, data3):
    # 降维
    d1 = data1.flatten()
    d2 = data2.flatten()
    d3 = data3.flatten()

    # 转为列的形式
    dd1 = d1[:, np.newaxis]
    dd2 = d2[:, np.newaxis]
    dd3 = d3[:, np.newaxis]

    data4 = np.c_[dd1, dd2, dd3]

    # 除去存在nan的行，过滤数据
    data5 = np.delete(data4, np.where(np.isnan(data4))[0], axis=0)

    return data5


def show_image(element, file, hy1cfilepath):
    x = np.size(element, 1)
    y = np.size(element, 0)
    fy4a_Lon_Size = x
    fy4a_Lat_Size = y

    # 自定义FY4A的南海范围经纬度，间隔为0.04°
    lon = np.linspace(103, 123, fy4a_Lon_Size)
    lat = np.linspace(25, 4, fy4a_Lat_Size)

    # 网格化
    [fy4a_grid_lon1, fy4a_grid_lat1] = np.meshgrid(lon, lat)

    fig, ax = plt.subplots(figsize=(8, 7))
    cf0 = ax.contourf(fy4a_grid_lon1, fy4a_grid_lat1, element, np.arange(12, 33, .5),
                      extend='both', cmap='Spectral_r')
    cb = plt.colorbar(cf0, )

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'fontproperties': 'Kaiti',
            'size': 16,
            }
    cb.set_label('温度C°', fontdict=font, fontproperties='Kaiti')  # 设置colorbar的标签字体及其大小
    ax.set_xlabel('经度', fontsize=16, fontdict=font, fontproperties='Kaiti')
    ax.set_ylabel('纬度', fontsize='x-large', fontdict=font, fontproperties='Kaiti')

    plt.tight_layout()
    plt.savefig(file, dpi=1000)


def Create_Ncfile(SST1, hy1c_lon1, hy1c_lat1, nc_Output_File):
    newfile = netCDF4.Dataset(nc_Output_File, 'w', format='NETCDF4')

    newfile.createdate = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    newfile.Statistic = 'daily'
    newfile.Map_Projection = 'Equal_lat_lon'
    newfile.Northernmost_Latitude = 25
    newfile.Southernmost_Latitude = 4
    newfile.Westernmost_Longitude = 103
    newfile.Easternmost_Longitude = 123
    newfile.Latitude_Step = 0.02
    newfile.Longitude_Step = 0.02
    newfile.Region = 'The south China sea'

    # 创建维度变量
    nx = np.size(hy1c_lon1, 0)
    ny = np.size(hy1c_lon1, 1)
    newfile.createDimension('dim_0', nx)
    newfile.createDimension('dim_1', ny)
    hy1c_lon = np.linspace(103, 123, ny)
    hy1c_lat = np.linspace(25, 4, nx)

    # 创建变量
    latitude = newfile.createVariable('lat', 'f4', ('dim_0', 'dim_1'))
    longitude = newfile.createVariable('lon', 'f4', ('dim_0', 'dim_1'))
    SST = newfile.createVariable('SST', 'f4', ('dim_0', 'dim_1'), fill_value=-999)

    # 设置变量属性
    latitude.units = 'Degrees'
    longitude.units = 'Degrees'

    SST.long_name = 'Sea surface temperature in the South China Sea'
    SST.units = 'C_Degrees'

    # 设置无效值

    SST1[np.isnan(SST1)] = -999
    # 设置变量值
    newfile.variables['lat'][:] = hy1c_lat1[:]
    newfile.variables['lon'][:] = hy1c_lon1[:]
    newfile.variables['SST'][:] = SST1[:]

    newfile.close()


def show_testimage(element, file, hy1cfilepath):
    x = np.size(element, 1)
    y = np.size(element, 0)
    fy4a_Lon_Size = x
    fy4a_Lat_Size = y

    # 自定义FY4A的南海范围经纬度，间隔为0.04°
    lon = np.linspace(103, 123, fy4a_Lon_Size)
    lat = np.linspace(25, 4, fy4a_Lat_Size)

    # 网格化
    [fy4a_grid_lon1, fy4a_grid_lat1] = np.meshgrid(lon, lat)

    fig, ax = plt.subplots(figsize=(8, 7))
    cf0 = ax.contourf(fy4a_grid_lon1, fy4a_grid_lat1, element, np.arange(10, 35, .5),
                      extend='both', cmap='jet')
    cb = plt.colorbar(cf0, )
    cb.set_label('colorbar', fontdict=font)
    # 获取日期
    # a = len(hy1cfilepath)
    # year = hy1cfilepath[a - 22:a - 18]
    # month =hy1cfilepath[a - 18:a - 16]
    # day = hy1cfilepath[a - 16:a - 14]
    cb = plt.colorbar(cf0, )

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 16,
            }
    cb.set_label('C°', fontdict=font)  # 设置colorbar的标签字体及其大小
    ax.set_xlabel('Longitude', fontsize=16, fontdict=font)
    ax.set_ylabel('Latitude', fontsize='x-large', fontdict=font)
    plt.savefig(file, dpi=1000)


def show_nanhai_hy1c(lon, lat, data, save_file):
    fig, ax = plt.subplots(figsize=(8, 7))
    cf0 = ax.contourf(lon, lat, data, np.arange(10, 35, .5),
                      extend='both', cmap='jet')
    cb = plt.colorbar(cf0, )

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 16,
            }
    cb.set_label('C°', fontdict=font)  # 设置colorbar的标签字体及其大小
    ax.set_xlabel('Longitude', fontsize=16, fontdict=font)
    ax.set_ylabel('Latitude', fontsize='x-large', fontdict=font)

    plt.tight_layout()
    plt.savefig(save_file, dpi=1000)


def Run_Data(fy4filepath, hy1cfilepath, outputpath):
    path = os.path.split(os.path.realpath(__file__))[0]
    mask1 = sio.loadmat(os.path.join(path, 'nanhai_mask01.mat'))
    mask2 = sio.loadmat(os.path.join(path, 'nanhai_mask02.mat'))
    mask3 = sio.loadmat(os.path.join(path, 'nanhai_mask03.mat'))

    # 读取sst要素
    in_Fy4_File = netCDF4.Dataset(fy4filepath, mode='r')
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
    kriging_Fy4a_SST = Kriging(fy4a_grid_lon2, fy4a_grid_lat2, original_Fy4a_SST, fy4a_Lon_Size, fy4a_Lat_Size)

    # 将插值的数据填充到缺失值处，分为两步：
    # step1：找到原始数据的nan的索引
    where_are_NaNs = np.isnan(original_Fy4a_SST)
    # step1：矩阵运算  把原始SST的值赋给新变量final_Fy4a_SST
    final_Fy4a_SST = original_Fy4a_SST
    final_Fy4a_SST[where_are_NaNs] = kriging_Fy4a_SST[where_are_NaNs]
    # 加载海南掩膜1  这个掩膜是间隔0.04

    mask_data1 = mask1['nanhai_mask01']
    # 陆地掩膜
    # 1代表陆地，0代表海洋
    final_Fy4a_SST[mask_data1 == 1] = np.nan

    # 这里是读取HY1C数据
    # 然后输出三列数据，依次是经度、纬度、海温

    # 读取sst要素
    in_file = h5py.File(hy1cfilepath, mode='r')
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
    # 克里金插值，得到hy1c的南海数据kriging_Hy1c_SST
    kriging_Hy1c_SST = Kriging(hy1c_lon2, hy1c_lat2, hy1c_SST1, hy1c_Lon_Size, hy1c_Lat_Size)

    # data1是风云四号数据，三列数据，依次是经度、纬度、海温
    data1 = Arranged_In_Three_Columns(fy4a_grid_lon2, fy4a_grid_lat2, final_Fy4a_SST)

    # 把final_Fy4a_SST插值为和hy1c具有相同网格大小的矩阵
    fy4a_mat = griddata(data1[:, 0:2], data1[:, 2], (hy1c_lon2, hy1c_lat2), method='linear')

    # final_Fy4a_SST与hy1c_SST1相加，再求平均
    fy4a_hy1c_mat = (np.array(fy4a_mat) + np.array(kriging_Hy1c_SST)) / 2
    # 加载海南掩膜2  这个掩膜是间隔0.03704

    mask_data2 = mask2['nanhai_mask02']

    # 掩膜
    # 1代表陆地，0代表海洋
    fy4a_hy1c_mat[mask_data2 == 1] = np.nan

    # 插值为0.02°间隔

    [output_data1, lon4, lat4] = Interpolate_To_002_Degrees(hy1c_lon2, hy1c_lat2, fy4a_hy1c_mat, 1001, 1051)

    # 加载海南掩膜3  这个掩膜是间隔0.02

    mask_data3 = mask3['nanhai_mask03']
    # 掩膜
    # 1代表陆地，0代表海洋
    output_data1[mask_data3 == 1] = np.nan

    # 获取 输出文件的路径、名字
    a = len(hy1cfilepath)
    name1 = hy1cfilepath[a - 22:a - 14]
    name2 = 'FY4A_HY1C_SST_' + name1 + '.nc'
    name3 = 'FY4A_HY1C_SST_' + name1 + '.png'
    nc_Output_File = os.path.join(outputpath, name2)  # 'D:\FY4A_HY1C_SST_20200826.nc'
    # 保存nc文件
    Create_Ncfile(output_data1, lon4, lat4, nc_Output_File)
    png_Output_File = os.path.join(outputpath, name3)
    # show_testimage(output_data1, r'D:\DDD\第四篇论文\图\FY4A_HY1C_SST', hy1c_file)
    # show_testimage(kriging_Fy4a_SST, r'D:\DDD\第四篇论文\图\kriging_Fy4a_SST', hy1c_file)
    # show_testimage(final_Fy4a_SST, r'D:\DDD\第四篇论文\图\final_Fy4a_SST', hy1c_file)
    # show_testimage(kriging_Hy1c_SST, r'D:\DDD\第四篇论文\图\kriging_Hy1c_SST', hy1c_file)

    # 保存图片
    show_image(output_data1, png_Output_File, hy1cfilepath)

    A = 1


def find_ncfile(path):
    nc_file = []
    for f_name in os.listdir(path):
        if f_name.endswith('.HDF'):
            # print(f_name)
            # nc_file = f_name
            nc_file.append(f_name)
    return nc_file


def find_h5file(path):
    h5_file = []
    for f_name in os.listdir(path):
        if f_name.endswith('.h5'):
            # print(f_name)
            # nc_file = f_name
            h5_file.append(f_name)
    return h5_file


def main(fy4path, hy1cpath, outputpath):
    fy4_list = find_ncfile(fy4path)
    for i in range(len(fy4_list)):
        fy4file = os.path.join(fy4path, fy4_list[i])
        date = fy4file[-27:-19]

        hy1c_list = find_h5file(hy1cpath)
        for j in range(len(hy1c_list)):
            if date in hy1c_list[j]:
                hy1c_file = os.path.join(hy1cpath, hy1c_list[j])
                print('FY4A file is :', fy4file)
                print('HY1C file is :', hy1c_file)
                Run_Data(fy4file, hy1c_file, outputpath)
            else:
                pass


def fy4aimage(fy4file, outputpath):

    # 读取sst要素
    in_Fy4_File = netCDF4.Dataset(fy4file, mode='r')
    # 这里是FY4A的变量，目前是固定的（如果后续发生命名变化，这里需要改）
    fy4a_var = 'Data'

    fy4a_SST = in_Fy4_File[fy4a_var][:].astype(np.float).copy()
    fy4a_SST1 = fy4a_SST
    fy4a_SST1 = np.where(fy4a_SST1 == np.min(fy4a_SST1), np.nan, fy4a_SST1)
    fy4a_SST1 = np.where(fy4a_SST1 == np.nanmax(fy4a_SST1), -888, fy4a_SST1)


    # 中国范围
    x_min = 11
    x_max = 55
    y_min = 74
    y_max = 136

    fy4a_Lon_Size = np.size(fy4a_SST1, 1)
    fy4a_Lat_Size = np.size(fy4a_SST1, 0)
    lon = np.linspace(y_min, y_max, fy4a_Lon_Size)
    lat = np.linspace(x_max, x_min, fy4a_Lat_Size)
    # 网格化
    [grid_lon1, grid_lat1] = np.meshgrid(lon, lat)

    fig, ax = plt.subplots(figsize=(8, 7))
    cf0 = ax.contourf(grid_lon1, grid_lat1,fy4a_SST1, np.arange(12, 33, .5),
                      extend='both', cmap='Spectral_r')
    cb = plt.colorbar(cf0, )
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'fontproperties': 'Kaiti',
            'size': 16,
            }
    cb.set_label('温度C°', fontdict=font, fontproperties='Kaiti')  # 设置colorbar的标签字体及其大小
    ax.set_xlabel('经度', fontsize=16, fontdict=font, fontproperties='Kaiti')
    ax.set_ylabel('纬度', fontsize='x-large', fontdict=font, fontproperties='Kaiti')

    plt.tight_layout()

    # 设置fy4的输出文件名
    basefile = os.path.basename(fy4file)[:-3] + 'png'
    file = os.path.join(outputpath, basefile)
    plt.savefig(file, dpi=1000)
    plt.close()
    # 南海FY4数据
    fy4a_SST2 = fy4a_SST[725:1251, 825:1326]
    # fy4a_SST2= np.flipud(fy4a_SST2)

    # 经纬度
    Lon_Size = np.size(fy4a_SST2, 1)
    Lat_Size = np.size(fy4a_SST2, 0)
    lon = np.linspace(103, 123, Lon_Size)
    lat = np.linspace(25, 4, Lat_Size)
    # 网格化
    [grid_lon1, grid_lat1] = np.meshgrid(lon, lat)


    fy4a_SST2 = np.where(fy4a_SST2 == np.nanmin(fy4a_SST2), np.nan, fy4a_SST2)
    fy4a_SST2 = np.where(fy4a_SST2 == np.nanmax(fy4a_SST2), -888, fy4a_SST2)
    # 设置nanhai_fy4a的输出文件名
    nanhai_basefile = os.path.basename(fy4file)[:-4] + 'hainan.png'
    nanhai_file= os.path.join(outputpath, nanhai_basefile)
    #画图
    fig, ax = plt.subplots(figsize=(8, 7))
    cf0 = ax.contourf(grid_lon1, grid_lat1,fy4a_SST2, np.arange(12, 33, .5),
                      extend='both', cmap='Spectral_r')
    cb = plt.colorbar(cf0, )
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'fontproperties': 'Kaiti',
            'size': 16,
            }
    cb.set_label('温度C°', fontdict=font, fontproperties='Kaiti')  # 设置colorbar的标签字体及其大小
    ax.set_xlabel('经度', fontsize=16, fontdict=font, fontproperties='Kaiti')
    ax.set_ylabel('纬度', fontsize='x-large', fontdict=font, fontproperties='Kaiti')

    plt.tight_layout()
    plt.savefig(nanhai_file, dpi=1000)
    plt.close()

def compare_image(fy4file, outputpath):

    # 读取sst要素
    in_Fy4_File = netCDF4.Dataset(fy4file, mode='r')
    # 这里是FY4A的变量，目前是固定的（如果后续发生命名变化，这里需要改）
    fy4a_var = 'Data'

    fy4a_SST = in_Fy4_File[fy4a_var][:].astype(np.float).copy()
    fy4a_SST1 = fy4a_SST

    fy4a_SST2 = fy4a_SST[725:1251, 825:1326]
    # 经纬度
    Lon_Size = np.size(fy4a_SST2, 1)
    Lat_Size = np.size(fy4a_SST2, 0)
    lon = np.linspace(103, 123, Lon_Size)
    lat = np.linspace(25, 4, Lat_Size)
    # 网格化
    [grid_lon1, grid_lat1] = np.meshgrid(lon, lat)


    fy4a_SST2 = np.where(fy4a_SST2 == np.nanmin(fy4a_SST2), np.nan, fy4a_SST2)
    fy4a_SST2 = np.where(fy4a_SST2 == np.nanmax(fy4a_SST2), -888, fy4a_SST2)

    # 'linear'
    # 'nearest'
    # 'cubic'
    method = 'cubic'
    data = Arranged_In_Three_Columns(grid_lon1, grid_lat1, fy4a_SST2)
    output_data1 = griddata(data[:, 0:2], data[:, 2], (grid_lon1, grid_lat1), method=method)
    #画图
    fig, ax = plt.subplots(figsize=(8, 7))
    cf0 = ax.contourf(grid_lon1, grid_lat1,output_data1, np.arange(12, 33, .5),
                      extend='both', cmap='Spectral_r')
    cb = plt.colorbar(cf0, )
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'fontproperties': 'Kaiti',
            'size': 16,
            }
    cb.set_label('温度C°', fontdict=font, fontproperties='Kaiti')  # 设置colorbar的标签字体及其大小
    ax.set_xlabel('经度', fontsize=16, fontdict=font, fontproperties='Kaiti')
    ax.set_ylabel('纬度', fontsize='x-large', fontdict=font, fontproperties='Kaiti')

    plt.tight_layout()

    # 设置nanhai_fy4a的输出文件名
    nanhai_basefile = os.path.basename(fy4file)[:-4] + method+'.jpg'
    nanhai_file= os.path.join(outputpath, nanhai_basefile)
    plt.savefig(nanhai_file, dpi=1000)
    plt.close()



def hy1cimage(hy1cfile, outputpath):
    path = os.path.split(os.path.realpath(__file__))[0]
    mask1 = sio.loadmat(os.path.join(path, 'nanhai_mask01.mat'))
    mask2 = sio.loadmat(os.path.join(path, 'nanhai_mask02.mat'))
    mask3 = sio.loadmat(os.path.join(path, 'nanhai_mask03.mat'))

    # 读取sst要素
    in_file = h5py.File(hy1cfile, mode='r')
    # 这里是HY1C的变量，目前是固定的（如果后续发生命名变化，这里需要改）
    var = '/Geophysical Data/SST'
    hy1c_SST = in_file[var][:].astype(np.float).copy()
    hy1c_SST1 = hy1c_SST[1755:2322, 7641:8181]
    hy1c_SST1 = np.where(hy1c_SST1 == np.min(hy1c_SST1), 100, hy1c_SST1)
    mask_data2 = mask2['nanhai_mask02']
    hy1c_SST1[mask_data2 == 1] = -888
    # hy1c_SST1 = np.flipud(hy1c_SST1)

    Lon_Size = np.size(hy1c_SST1, 1)
    Lat_Size = np.size(hy1c_SST1, 0)
    lon = np.linspace(103, 123, Lon_Size)
    lat = np.linspace(25, 4, Lat_Size)
    # 网格化
    [grid_lon1, grid_lat1] = np.meshgrid(lon, lat)


    # 陆地设置为-888，缺失值设置为nan

    hy1c_SST1 = np.where(hy1c_SST1 == 100,np.nan , hy1c_SST1)
    fig, ax = plt.subplots(figsize=(8, 7))
    cf0 = ax.contourf(grid_lon1, grid_lat1,hy1c_SST1, np.arange(12, 33, .5),
                      extend='both', cmap='Spectral_r')
    cb = plt.colorbar(cf0, )
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'fontproperties': 'Kaiti',
            'size': 16,
            }
    cb.set_label('温度C°', fontdict=font, fontproperties='Kaiti')  # 设置colorbar的标签字体及其大小
    ax.set_xlabel('经度', fontsize=16, fontdict=font, fontproperties='Kaiti')
    ax.set_ylabel('纬度', fontsize='x-large', fontdict=font, fontproperties='Kaiti')

    plt.tight_layout()

    # 设置hy1c的输出文件名
    basefile = os.path.basename(hy1cfile)[:-2] + 'png'
    file = os.path.join(outputpath, basefile)
    plt.savefig(file, dpi=1000)
    plt.close()

def make_image(fy4path, hy1cpath, outputpath):
    fy4_list = find_ncfile(fy4path)
    for i in range(len(fy4_list)):
        fy4file = os.path.join(fy4path, fy4_list[i])
        date = fy4file[-27:-19]

        hy1c_list = find_h5file(hy1cpath)
        for j in range(len(hy1c_list)):
            if date in hy1c_list[j]:
                hy1c_file = os.path.join(hy1cpath, hy1c_list[j])
                print('FY4A file is :', fy4file)
                print('HY1C file is :', hy1c_file)
                fy4aimage(fy4file, outputpath)
                hy1cimage(hy1c_file, outputpath)
            else:
                pass

if __name__ == "__main__":
    fy4path = r'D:\github\FY4_data\fy4'
    hy1cpath = r'D:\github\FY4_data\hy1c'
    outputpath = r'D:\github\FY4_data\out'
    # main(fy4path, hy1cpath, outputpath)

    fy4file = r'D:\github\FY4_data\fy4\FY4A_AGRIX_L3_GLL_20200925_POAD_4000M_SST.HDF'
    hy1cfile = r'D:\github\FY4_data\hy1c\H1C_OPER_OCT_L3A_20200925_SST_4KM_13.h5'
    outputpath = r'D:\github\FY4_data\out'
    hy1cout='D:\github\FY4_data\hy1c_out'
    fy4cout = r'D:\github\FY4_data\fy4a_out'
    hy1cimage(hy1cfile,hy1cout)
    fy4aimage(fy4file,fy4cout)
    # Run_Data(fy4file, hy1c_file, outputpath)
