# coding:utf-8

# Create your tests here.

from multiprocessing import Pool


def f(x):
    for i in range(1):
        x *= x
    return x


if __name__ == '__main__':  # start 4 worker processes
    with Pool(processes=4) as pool:
        print(pool.map(f, range(1)))
        for i in pool.imap_unordered(f, range(1)):
            print(i)

# print same numbers in arbitrary order
# evaluate "f(20)" asynchronously res = pool.apply_async(f, (20,)) print(res.get(timeout=1))
# runs in *only* one process # prints "400"
# evaluate "os.getpid()" asynchronously res = pool.apply_async(os.getpid, ()) # runs in *only* one process print(res.get(timeout=1))
# prints the PID of that process
# launching multiple evaluations asynchronously *may* use more processes multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
