# CHEATSHEET

## 做递归防超时定式

```python
import sys
from functools import lru_cache
sys.setrecursionlimit(1 << 30)

@lru_cache(maxsize = None)
# def...
# 上面这一行必须和def紧挨着
# lru_cache仅适用于不可变对象

```

## 一些散装函数

### round()函数
```python
print(round(3.35))
print(round(3.35, 1))
'''
输出：
3
3.4
'''
```

第二个参数表示小数位数。

但是要注意，这样可能会舍去末尾的0，因此更建议如下操作：

```python
print('%.2f' % 9.801)
print('%.3f' % 15.50549)
'''
输出：
9.80
15.505
'''

```

### extend()方法
```python
lst = [1, 2]
lst.extend([3, 4])
lst.extend((5, 6))
lst.extend({7: 'seven', 8: 'eight'})
lst.extend({10, 9})
print(lst)
# 输出：[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### enumerate()函数
```python
names = ['Alice', 'Bob', 'Carl']
print(list(enumerate(names)))
print(list(enumerate(names, 1)))
'''
输出：
[(0, 'Alice'), (1, 'Bob'), (2, 'Carl')]
[(1, 'Alice'), (2, 'Bob'), (3, 'Carl')]
'''
```

### eval()函数

将字符串解析为表达式。

```python
from math import sqrt
print(eval('5 * 3 + sqrt(2)'))
# 输出：16.414213562373096

```


## collections库

### Counter

```python
from collections import Counter

nums = [1, 1, 1, 6, 6, 6, 7, 8]
c = Counter(nums)
for k, v in c.items():
    print(k, v)
'''
输出：
1 3
6 3
7 1
8 1
'''

print(c)
# 输出：Counter({1: 3, 6: 3, 7: 1, 8: 1})

ansdict = c.most_common(2)
print(ansdict)
# 输出：[(1, 3), (6, 3)]

```

### defaultdict

```python
s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
d = defaultdict(list)
for k, v in s:
    d[k].append(v)

print(d)
print(sorted(d.items()))
print(d['green'])
'''
输出：
defaultdict(<class 'list'>, {'yellow': [1, 3], 'blue': [2, 4], 'red': [1]})
[('blue', [2, 4]), ('red', [1]), ('yellow', [1, 3])]
[]
'''

s = 'mississippi'
d = defaultdict(int)
for k in s:
    d[k] += 1
print(d)
print(d['a'])
'''
输出：
defaultdict(<class 'int'>, {'m': 1, 'i': 4, 's': 4, 'p': 2})
0
'''


```


## itertools库

### permutations

```python
from itertools import permutations
a = 'abc'   # 对字符串进行permutations排列组合
for i in permutations(a, 3):
    x = ''.join(i)
    print(x, end = ' ')
# 输出：abc acb bac bca cab cba

c = ('e', 'f', 'g')  # 对元组进行permutations排列组合
for j in permutations(c, 2):
    print(j)
'''
输出：
('e', 'f')
('e', 'g')
('f', 'e')
('f', 'g')
('g', 'e')
('g', 'f')
'''
```

## bisect库

### bisect(), bisect_left(), bisect_right()

输入的列表应有序！

bisect()和bisect_right()等同。bisect_left()和bisect_right()分别返回大于等于和大于指定值的第一个索引。

```python
from bisect import *
ls = [1, 3, 5, 5, 5, 7, 7, 9]
print(bisect_left(ls, 5))
print(bisect_right(ls, 5))
print(bisect(ls, 5))
print(bisect(ls, 15))

'''
输出：
2
5
5
8
'''

```

## sys库

利用stdin.read()可以一次读取多行数据。若有如下程序：

```python
import sys
data = sys.stdin.read().strip()
datas = data.split('\n')

for x in datas:
    print(int(x) + 1)

```

若输入：

```python
113
513
```

在本地不会直接输出，但在做题网站上会输出：

```python
114
514
```

### 埃氏筛

```python
prime = [True for _ in range(n + 1)]
primes = []
p = 2
while p * p <= n:
    if prime[p]:
        primes.append(p)
        for i in range(p * p, n + 1, p):
            prime[i] = False
    p += 1
```

### 欧拉筛

```python
primes = []
is_prime = [True] * (n + 1)
for i in range(2, n + 1):
    if is_prime[i]:
        primes.append(i)
    for j in primes:
        if i * j > n:
            break
        is_prime[i * j] = False
        if i % j == 0:
            break
```

## math库

```python
import math
print(math.gcd(90, 54))
print(math.floor(1.14))
print(math.ceil(1.14))
print(math.log(729, 3))
print(math.factorial(5))
print(math.isclose(0.1 + 0.2, 0.3))

'''
输出：
18
1
2
6.0
120
True
'''
```

### Dilworth定理

最少严格/不严格上升子序列的分割数=最长不严格/严格下降子序列长度。

```python
import bisect
arr = [9, 4, 10, 5, 1]
arr.reverse()
a = []
for x in arr:
    idx = bisect.bisect(a, x)
    if idx == len(a):
        a.append(x)
    else:
        a[idx] = x
print(len(a))
# 输出：3
```

### 二分查找

valid(i) == True的i值有上界。

```python
def valid(i):
    return i < 114514

left, right = 0, 10**8
# left是左边界，right是右边界 + 1
while left < right:
    mid = (left + right) // 2
    if valid(mid):
        left = mid + 1
    else:
        right = mid
print(left - 1)
#输出：114513
```
