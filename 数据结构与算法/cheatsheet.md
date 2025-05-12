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

### 滑动窗口

```python
n = len(nums)
right = 0
for left in range(n):
    # 做与left有关的操作
    while right < n and (与right有关的某一条件):
        # 做与right有关的操作
        right += 1
    # 做一些操作，如增加计数等，比如n - right + 1
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

### KMP算法

用于寻找字符串`text`中出现字符串模式`pattern`的位置。

```python
def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    for i in range(1, m):
        while length > 0 and pattern[i] != pattern[length]:
            length = lps[length - 1]
        if pattern[i] == pattern[length]:
            length += 1
        lps[i] = length
    return lps

def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    if m == 0:
        return []
    lps = compute_lps(pattern)
    matches = []

    j = 0
    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i - j + 1)
            j = lps[j - 1]
    return matches

print(kmp_search('ABABABABCABABABABCABABABABC', 'ABABCABAB'))
# [4, 13]
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

另外，对于单调递增栈，每当处理完一个索引，单调栈内的某一个索引所对应的元素，就是该索引到栈中下一个索引在数组中对应所有元素的最小值。对单调递减栈同理。

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

### 树

二叉树的定义

```python
class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

多叉树的定义

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.children = []
```

#### 二叉搜索树的建立

```python
def insert(value, node):
    if value == node.val:
        return
    if value < node.val:
        if node.left is None:
            node.left = TreeNode(value)
        else:
            insert(value, node.left)
    else:
        if node.right is None:
            node.right = TreeNode(value)
        else:
            insert(value, node.right)
```

#### 二叉搜索树的验证

第一种方式是验证中序遍历序列是否是严格递增序列。

```python
def isBST(root):
    stack, cur = [], float('-inf')
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        if root.val <= cur:
            return False
        cur = root.val
        root = root.right

    return True
```

第二种方式是验证节点值是否在合法范围内。

```python
def helper(node, lower=float('-inf'), upper=float('inf')):
    if node is None:
        return True
    value = node.val
    if value <= lower or value >= upper:
        return False
    return helper(node.left, lower, value) and helper(node.right, value, upper)
```

#### Huffman编码

目的：用二叉树的叶节点存储字符，并最小化叶节点与叶节点权值之积的总和。

思路：弹出堆中最小的两个节点，合并并入堆。循环往复，直至堆中只剩下根节点。

```python
from heapq import heappop, heappush, heapify

class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.val < other.val


n = 4
heap = [TreeNode(i) for i in [1, 1, 3, 5]
heapify(heap)

for _ in range(n - 1):
    left_node, right_node = heappop(heap), heappop(heap)
    merged = TreeNode(left_node.val + right_node.val)
    merged.left, merged.right = left_node, right_node
    heappush(heap, merged)

stack = [(heap[0], 0)]
ans = 0
while stack:
    node, depth = stack.pop()
    if node.left is None and node.right is None:
        ans += node.val * depth
        continue
    if node.right is not None:
        stack.append((node.right, depth + 1))
    if node.left is not None:
        stack.append((node.left, depth + 1))

print(ans)
# 17
```

#### 多叉树的表示——长子-兄弟表示法

一个节点的左指针为其第一个子节点，右指针为其下一个兄弟节点。因此，前序遍历序列不变。

#### 并查集

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

### 图

#### 拓扑排序——Kahn算法

对无环有向图进行拓扑排序，对于任意边`(u, v)`，排序结果`result`中`u`都在`v`前面。若无法把所有顶点加入`result`中，则有环。

```python
from collections import deque, defaultdict

def topological_sort(graph):
    degree = defaultdict(int)
    result = []

    for u in graph:
        for v in graph[u]:
            degree[v] += 1

    q = deque(u for u in graph if degree[u] == 0)
    while q:
        u = q.popleft()
        result.append(u)
        for v in graph[u]:
            degree[v] -= 1
            if degree[v] == 0:
                q.append(v)

    return result if len(result) == len(graph) else None

def main():
    graph = {
        'A': ['B', 'C'],
        'B': ['C', 'D'],
        'C': ['E'],
        'D': ['F'],
        'E': ['F'],
        'F': []
    }
    print(topological_sort(graph))

main()
# ['A', 'B', 'C', 'D', 'E', 'F']
```

#### 最小生成树——Prim算法&Kruskal算法

找到一棵连接所有`n`个节点的包含`n - 1`条边的树，它在所有这样的树中权值之和最小。

Prim算法：选定某一起始节点，不断选择已生成的树通往外部的边中权值最小的一条，将其加入树中。


Kruskal算法：对所有边`edges`按权值进行排序，遍历每一条边，利用并查集，如果一条边的两个节点尚未在同一个连通分量中，则将该边加入结果`result`中。

```python
# class DisjointSet:
#     ...

djs = DisjointSet(n)
result = []
for w, u, v in sorted(edges):
    if djs.find(u) != djs.find(v):
        djs.union(u, v)
        result.append((w, u, v))
```

## 语法&小技巧

### ASCII

```python
print(ord('A'), ord('a'), chr(65))
# 65 97 A
```

### 保留小数位数

```python
print('%.5f' % 2 ** 0.5)
# 1.41421
num = 1.145141919810
print(f'{num:.7f}') # 不能随便加空格！
# 1.1451419
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


