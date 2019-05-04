from bypy import ByPy
import os
import time
import datetime
import threading
 
# 百度云存放文件的文件夹名
dir_name = "ByPy-test"
 
# 获取一个bypy对象，封装了所有百度云文件操作的方法
bp = ByPy()
# 百度网盘创建远程文件夹bypy-test
bp.mkdir(remotepath = dir_name)
 
# 函数作用：文件中的 \ 改为 /
# 函数输入：文件绝对路径
# 输出：文件绝对路径添加转义符后的结果
def changePath(filePath):
	path = ""
	for i in range(len(filePath)):
		if filePath[i] != "\\":
			path += filePath[i]
		else:
    		 path += "/"
	return path
 
# 根据当前路径和文件夹路径得到相对路径
def relPath(filePath, topDir):
	relativepath = ""
	for i in range(len(filePath)):
		if i < len(topDir) and filePath[i] == topDir[i]:
			continue
		relativepath += filePath[i]
	#print ("相对路径" + relativepath)
	return relativepath
 
# 函数作用：给出文件夹，得到所有文件的绝对路径
# 输入参数：当前文件夹的绝对路径
# 返回值：一个包含所有文件绝对路径,以及文件所在文件夹的大小的列表
def getFileList(file_dir):    
	fileList = []
	top_dir = ""
	checkFlag = False
	for root, dirs, files in os.walk(file_dir):
		#print(root) #当前目录路径  
		if checkFlag == False:
			top_dir = root
			checkFlag = True		
		#print(dirs) #当前路径下所有子目录  
		#print(files) #当前路径下所有非目录子文件  
		for file in files: 
			fileDict = dict(Path = changePath(relPath(root, top_dir)), fileName = file, createFlag = False)
			fileList.append(fileDict) # 当前目录+文件名
			#print(fileDict)
	return fileList
 
#获取文件的大小,结果保留两位小数，单位为MB
def get_FileSize(filePath):
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024*1024)
    return round(fsize,2)
 
# 获取文件绝对路径列表
allFiles = getFileList(os.path.abspath('.'))
 
totalFileSize = 0  # 文件大小变量
 
start = datetime.datetime.now()   # 计时开始
 
# 逐个上传
createFlag = {}
for file in allFiles:
	#bp.upload(localpath=file, remotepath=dir_name, ondup='newcopy')
	print("正在上传文件:" + file["fileName"])
	
	if file["Path"] != "":
		bp.mkdir(remotepath = dir_name + file["Path"])
		DIR_NAME = dir_name +  file["Path"]
		bp.upload(localpath= "." + file["Path"]+ "/" +file["fileName"], remotepath = str(DIR_NAME), ondup='newcopy')
		print ("文件发送完成：本地路径：" +  file["Path"]+"/" +file["fileName"] + " 远程文件夹：" + DIR_NAME)
		totalFileSize += get_FileSize( "." + file["Path"]+ "/" +file["fileName"])
	else:		
		bp.upload(localpath= file["fileName"], remotepath= dir_name, ondup='newcopy')
		print ("文件发送完成：" + file["fileName"] + " 远程文件夹：" + dir_name)
		totalFileSize += get_FileSize( "." + file["Path"]+ "/" +file["fileName"])
	print ("------------------------------------")
end = datetime.datetime.now()  # 计时结束
 
print("上传文件总大小为" + str(totalFileSize) + "MB")
print("花费时间(s)：" + str((end - start).seconds))
print("\nupload ok")