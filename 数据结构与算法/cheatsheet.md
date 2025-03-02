# CHEETSHEAT

## 算法

### 二分查找

```python
# small, large指可能取到的边界值
# valid(x)大于等于某一值时为True，反之为False
def binary_search_greatest_lower_bound(small, large):
    left, right = small, large
    while left < right:
        mid = (left + right) // 2
        if valid(mid):
            right = mid
        else:
            left = mid + 1
    return left

# valid(x)小于等于某一值时为True，反之为False
def binary_search_least_upper_bound(small, large):
    left, right = small, large + 1
    while left < right:
        mid = (left + right) // 2
        if valid(mid):
            left = mid + 1
        else:
            right = mid
    return right - 1
```

## 数据结构

### 链表

### 二叉树

### 图

## 语法&小技巧

### 保留小数位数

```python
print('%.5f' % 2 ** 0.5)
# 1.41421
```

### 字符串解析为表达式

```python
from math import sqrt
print(eval('5 * 3 + sqrt(2)'))
# 16.414213562373096
```

### 全排列

```python
from itertools import permutations
a = 'abc'
for i in permutations(a):
    x = ''.join(i)
    print(x, end = ' ')
# abc acb bac bca cab cba

c = ('e', 'f', 'g')
for j in permutations(c, 2):
    print(j)
'''
('e', 'f')
('e', 'g')
('f', 'e')
('f', 'g')
('g', 'e')
('g', 'f')
'''
```

### 素数筛法

```python
# 埃氏筛
prime = [True] * (n + 1)
primes = []
p = 2
while p * p <= n:
    if prime[p]:
        primes.append(p)
        for i in range(p * p, n + 1, p):
            prime[i] = False
    p += 1

# 欧拉筛
primes = []
prime = [True] * (n + 1)
for i in range(2, n + 1):
    if prime[i]:
        primes.append(i)
    for j in primes:
        if i * j > n:
            break
        prime[i * j] = False
        if i % j == 0:
            break
```
