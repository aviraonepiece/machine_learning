#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tkinter as tk

window=tk.Tk()
window.title('用户异常行为检测')
window.geometry('1200x500')

a=tk.StringVar()
def do_check():
    window.update()
    a.set('0')
    print(a.get())

    label1 = tk.Label(window, text='机器学习算法：KNN近邻算法；       特征提取方法：最(不)频繁的操作', bg='green', font=('Arial', 12), width=150,
                      height=1)
    label2 = tk.Label(window, textvariable=a, bg=('green' if a.get() == '0' else 'red'), font=('Arial', 10))


    label1.pack()
    label2.pack()



b=tk.Button(window,text='检 测',command=do_check)
b.pack()

# def do_check():
#     a.set('sss')
#     return 5


e=tk.Entry(window,show=None)



window.mainloop()