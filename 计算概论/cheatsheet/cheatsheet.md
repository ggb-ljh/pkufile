# CHEATSHEET

## round()函数
round( x [, n] )，n表示小数位数。
## collections库
### Counter
Counter的用法示例：
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
