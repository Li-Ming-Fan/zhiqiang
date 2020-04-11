import multiprocessing as mp

class Model:
    def __init__(self):
        self.a = 1
        
global A                                     # 将需要在子进程中使用的自定义类对象申明为全局变量
def process_job(x):
    print("subProcess-a:",A.a)             # 在子进程中访问 Model类对象
    return x

if __name__ == '__main__':
    m = Model()
    global A
    A = m                                  # 对全局变量进行赋值
    pool = mp.Pool(5)
    res = pool.map(process_job, range(5))  # 开启子进程
    pool.close()
    pool.join()
    print(res)

