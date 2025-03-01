# CHEETSHEAT

## 算法

### 二分查找

```python
# small, large指可能取到的边界值
# valid(x)大于等于某一值时为True，反之为False
def binary_search_with_infimum(small, large):
    left, right = small, large
    while left < right:
        mid = (left + right) // 2
        if valid(mid):
            right = mid
        else:
            left = mid + 1
    return left

# valid(x)小于等于某一值时为True，反之为False
def binary_search_with_supremum(small, large):
    left, right = small, large + 1
    while left < right:
        mid = (left + right) // 2
        if valid(mid):
            left = mid + 1
        else:
            right = mid
    return right - 1




```
