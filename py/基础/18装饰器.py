#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''--------------------------------------------------
@Project -> File   ：pylearn -> 18装饰器
@Author ：Tecypus
@Date   ：2021/8/16 10:30
@Desc   ：
--------------------------------------------------'''

class Foo(object):
    def __init__(self, func):
        self._func = func

    def __call__(self):
        print ('class decorator runing')
        self._func()
        print ('class decorator ending')

@Foo
def bar():
    print ('bar')

bar()