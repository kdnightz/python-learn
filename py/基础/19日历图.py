#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''--------------------------------------------------
@Project -> File   ：pylearn -> 19日历图
@Author ：Tecypus
@Date   ：2021/8/18 9:18
@Desc   ：
--------------------------------------------------'''
import calendar
from datetime import date
mydate = date.today()
year_calendar_str = calendar.calendar(2021)
print(f"{mydate.year}年的日历图：{year_calendar_str}\n")