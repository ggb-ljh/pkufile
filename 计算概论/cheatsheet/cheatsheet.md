# CHEATSHEET

## 一些散装函数

### round()函数
```python
print(round( 3.35 ))
print(round(3.35,1))
'''
输出：
3
3.4
```
n表示小数位数。

### enumerate()函数
```python
names = ['Alice', 'Bob', 'Carl']
print(list(enumerate(names)))
print(list(enumerate(names,1)))
'''
输出：
[(0, 'Alice'), (1, 'Bob'), (2, 'Carl')]
[(1, 'Alice'), (2, 'Bob'), (3, 'Carl')]
```

## collections库

### Counter
Counter用法示例：
```python
from collections import Counter

nums=[1,1,1,6,6,6,7,8]
c=Counter(nums)
for k,v in c.items():
    print(k,v)
'''输出：
1 3
6 3
7 1
8 1'''

print(count)
#输出：Counter({1: 3, 6: 3, 7: 1, 8: 1})

ansdict=c.most_common(2)
print(ansdict)
#输出：[(1, 3), (6, 3)]

```

### defaultdict
defaultdict用法示例：

```python



```


## itertools库

### permutations
permutations用法示例：

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
'''输出：
('e', 'f')
('e', 'g')
('f', 'e')
('f', 'g')
('g', 'e')
('g', 'f')
'''
```


