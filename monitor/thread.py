#!/usr/bin/python
# -*- coding: UTF-8 -*-

import threading
import time
import os
from bypy import ByPy


def upload_file_rename(localpath,remotepath):
    bp.upload(localpath,remotepath,ondup='newcopy')
    print("upload ok:"+localpath)
    foldername=localpath.split('\\')[-1]
    path_name = localpath.split('\\')[:-1]
    newfile =""
    for f in path_name:
    	newfile = newfile+f+"\\"
    newfile = newfile +"up_"+foldername
    os.rename(localpath,newfile)
    print("rename ok:"+newfile)


def file_name(user_dir):
    file_list = list()
    for root, dirs, files in os.walk(user_dir):
        for file in files:
        	if file.startswith("up_")==False:
	            file_list.append(os.path.join(root, file))
    return file_list

class myThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self, threadID, name, counter,file_path):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.file_path = file_path
    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        while True:
	        time.sleep(3)
	        file_list = ''
	        file_list = file_name(self.file_path)
	        print("..................\n")
	        for file in file_list:
	        	 upload_file_rename(file,dir_name)


if __name__ == '__main__':
	dir_name = "save_image"
	bp = ByPy()
	bp.mkdir(remotepath = dir_name)
	# 创建新线程
	thread1 = myThread(1, "Thread-1", 1,"D:\\ai_projects\\monitor\\save_image")

	# 开启线程
	thread1.start()

	print("Exiting Main Thread")