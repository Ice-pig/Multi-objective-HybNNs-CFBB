import psutil


def notice():
    info = psutil.virtual_memory()
    k=1024*1024*1024
    used_memory = info.used / k
    total_memory = info.total / k
    memory_percent = info.percent
    cpu_num = psutil.cpu_count()

    print(u'used_memory/GB：',used_memory)
    print(u'total_memory/GB：',total_memory)
    print(u'percent：',info.percent)
    print(u'cpu个数：',psutil.cpu_count())
    return()


def memery_info():

    info = psutil.virtual_memory()
    k = 1024 * 1024 * 1024
    used_memory = info.used / k
    total_memory = info.total / k
    memory_percent = info.percent
    cpu_num = psutil.cpu_count()

    print(u'used_memory/GB：', used_memory)
    print(u'total_memory/GB：', total_memory)
    print(u'percent：', info.percent)
    print(u'cpu个数：', psutil.cpu_count())
    out = [used_memory, total_memory, memory_percent]
    return (out)