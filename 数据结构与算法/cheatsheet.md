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

## 语法

### 保留小数位数

```python
print('%.5f' % 2 ** 0.5)
# 1.41421
```
