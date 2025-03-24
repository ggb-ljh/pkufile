# CHEETSHEAT

## 算法

### 二分查找

`small`, `large`为可能取到的边界值。

`valid(x)`大于等于某一值时为`True`，反之为`False`的情况：

```python
def binary_search_greatest_lower_bound(small, large):
    left, right = small, large
    while left < right:
        mid = (left + right) // 2
        if valid(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

`valid(x)`小于等于某一值时为`True`，反之为`False`的情况：

```python
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

### 求排列的逆序数

```python
def merge_two(left, right):
    res = []
    cnt = 0
    p1 = p2 = 0
    l1, l2 = len(left), len(right)
    while p1 < l1 and p2 < l2:
        if left[p1] <= right[p2]:
            res.append(left[p1])
            p1 += 1
        else:
            res.append(right[p2])
            p2 += 1
            cnt += l1 - p1
    res.extend(left[p1:])
    res.extend(right[p2:])

    return res, cnt

def merge_self(nums):
    l = len(nums)
    if l == 1:
        return nums, 0
    mid = l >> 1
    merged_left, cnt_left = merge_self(nums[:mid])
    merged_right, cnt_right = merge_self(nums[mid:])
    merged_two, cnt = merge_two(merged_left, merged_right)

    return merged_two, cnt + cnt_right + cnt_left

seq = [2, 6, 3, 4, 5, 1]
_, ans = merge_self(seq)

print(ans)
# 8
```

## 数据结构

### 栈

#### 中缀表达式转后缀表达式——调度场算法

初始化运算符栈`operator`和输出栈`result`为空。对于表达式`tokens`的每个`token`：

如果`token`为数字：将其压入输出栈。

如果`token`为左括号：将其压入运算符栈。

如果`token`为右括号：将运算符栈顶元素弹出并压入输出栈，直到遇到左括号为止。

如果`token`为运算符：将运算符栈顶元素弹出并压入输出栈，直到运算符栈顶元素优先级小于`token`或遇到左括号为止。将`token`压入运算符栈。

将运算符栈顶元素弹出并压入输出栈，直到运算符栈为空为止。

```python
priority = {'(': 0,
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            }

# (3)*((3+4)*(2+3.5)/(4+5))
tokens = ['(', '3', ')', '*', '(', '(', '3', '+', '4', ')', '*', '(', '2', '+', '3.5', ')', '/', '(', '4', '+', '5', ')', ')']
operator = []
result = []
for token in tokens:
    if token in '+-*/':
        while operator and priority[operator[-1]] >= priority[token]:
            result.append(operator.pop())
        operator.append(token)
    elif token == '(':
        operator.append(token)
    elif token == ')':
        while operator[-1] != '(':
            result.append(operator.pop())
        operator.pop()
    else:
        result.append(token)
while operator:
    result.append(operator.pop())

print(*result)
# 3 3 4 + 2 3.5 + * 4 5 + / *
```

#### 单调栈

找到某个数组中，每个元素右边第一个比其更大的数的索引。这时使用单调递减栈。

```python
def find_next_greater(nums):
    n = len(nums)
    res = [0] * n
    stack = []

    for i in range(n):
        while stack and nums[i] > nums[stack[-1]]:
            res[stack.pop()] = i
        stack.append(i)

    while stack:
        res[stack.pop()] = n

    return res

print(find_next_greater([4, 5, 2, 25]))
# [1, 3, 3, 4]
```

### 链表

```python
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
```

#### 反转链表

反转链表并返回新的头节点。

```python
def reverse_linked_list(head):
    pre, cur = None, head
    while cur is not None:
        cur.next, pre, cur = pre, cur, cur.next
    return pre
```

#### 合并两个升序链表

合并两个升序链表并返回新的头节点。

```python
def merge_two_lists(head1, head2):
    dummy = ListNode(0)
    cur = dummy

    while head1 is not None and head2 is not None:
        if head1.val <= head2.val:
            cur.next, head1 = head1, head1.next
        else:
            cur.next, head2 = head2, head2.next
        cur = cur.next
    cur.next = head2 if head1 is None else head1
    
    return dummy.next
```

#### 查找链表中间节点

返回链表的中间节点或中间偏左节点。

```python
def find_middle_node(head):
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

### 并查集

注意可能根据题目需求不同，`union()`的实现方式需要调整。

```python
class DisjointSet:
    def __init__(self, k):
        self.parents = list(range(k))
        self.rank = [1] * k

    def find(self, x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def union(self, x, y):
        x_rep, y_rep = self.find(x), self.find(y)
        if x_rep == y_rep:
            return
        if self.rank[x_rep] < self.rank[y_rep]:
            self.parents[x_rep] = y_rep
        elif self.rank[x_rep] > self.rank[y_rep]:
            self.parents[y_rep] = x_rep
        else:
            self.parents[y_rep] = x_rep
            self.rank[x_rep] += 1
```




### 二叉树

### 图

## 语法&小技巧

### ASCII

```python
print(ord('A'))
# 65

print(chr(65))
# A
```

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
    print(x, end=' ')
# abc acb bac bca cab cba

c = ('e', 'f', 'g')
for j in permutations(c, 2):
    print(j)
# ('e', 'f')
# ('e', 'g')
# ('f', 'e')
# ('f', 'g')
# ('g', 'e')
# ('g', 'f')
```

### 正则表达式

`^`：匹配开始位置。

`$`：匹配结束位置。

`\`：匹配特殊字符，有`^$()*+?.\[{|`。

`*`：匹配前面的零次或多次。

`+`：匹配前面的一次或多次。

`?`：匹配前面的零次或一次。

`|`：或。

`\d`/`\D`：匹配阿拉伯数字/非阿拉伯数字。

`\w`/`\W`：匹配字母、数字、下划线/非字母、数字、下划线。

`\s`/`\S`：匹配空白/非空白。

`[]`：匹配范围内的字符之一，如`[ace]`可匹配`a`，`[m-p]`可匹配`o`。

`[^]`：匹配非范围内的字符。

`()`：将其中的内容作为整体。

```python
import re

reg = r'^(0|[1-9][0-9]*)$'
s = '26'
print('yes' if re.match(reg, s) else 'no')
# yes
```

### 素数筛法

#### 埃氏筛

```python
prime = [True] * (n + 1)
primes = []
p = 2
while p * p <= n:
    if prime[p]:
        primes.append(p)
        for i in range(p * p, n + 1, p):
            prime[i] = False
    p += 1
```

#### 欧拉筛

```python
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
