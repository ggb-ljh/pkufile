# CHEATSHEET

## 做递归防超时定式

```python
import sys
from functools import lru_cache
sys.setrecursionlimit(1 << 30)

@lru_cache(maxsize=None)
def #上面这一行必须和def紧挨着

```

## 一些散装函数

### round()函数
```python
print(round(3.35))
print(round(3.35,1))
'''
输出：
3
3.4
'''
```
第二个参数表示小数位数。

但是要注意，这样可能会舍去末尾的0，因此更建议如下操作：
```python
print('%.2f'%9.801)
print('%.3f'%15.50549)
'''
输出：
9.80
15.505
'''

```

### extend()方法
```python
lst=[1,2]
lst.extend([3,4])
lst.extend((5,6))
lst.extend({7:'seven',8:'eight'})
lst.extend({10,9})
print(lst)
#输出：[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### enumerate()函数
```python
names = ['Alice', 'Bob', 'Carl']
print(list(enumerate(names)))
print(list(enumerate(names,1)))
'''
输出：
[(0, 'Alice'), (1, 'Bob'), (2, 'Carl')]
[(1, 'Alice'), (2, 'Bob'), (3, 'Carl')]
'''
```

## collections库

### Counter

```python
from collections import Counter

nums=[1,1,1,6,6,6,7,8]
c=Counter(nums)
for k,v in c.items():
    print(k,v)
'''
输出：
1 3
6 3
7 1
8 1
'''

print(count)
#输出：Counter({1: 3, 6: 3, 7: 1, 8: 1})

ansdict=c.most_common(2)
print(ansdict)
#输出：[(1, 3), (6, 3)]

```

### defaultdict

```python
s  = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
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
from itertools import  permutations
a = 'abc'   #对字符串进行permutations排列组合
for i in permutations(a,3):
    x = ''.join(i)
    print (x,end=' ')
#输出：abc acb bac bca cab cba

c = ('e','f','g')  #对元组进行permutations排列组合
for j in permutations(c,2):
    print (j)
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
ls = [1,3,5,5,5,7,7,9]
print(bisect_left(ls,5))
print(bisect_right(ls,5))
print(bisect(ls,5))
print(bisect(ls,15))

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
data=sys.stdin.read().strip()
datas=data.split('\n')

for x in datas:
    print(1+int(x))

```

若输入：

```python
113
513
```

在本地似乎不会有输出，但在做题网站上会输出：

```python
114
514
```


