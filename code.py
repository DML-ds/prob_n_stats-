### Task 1
##### _using my student_id..._
import numpy as np
student_no = [2,3,4,7,8,7,6]
x = np.array(student_no)
x
#assign the sum of x to rs
rs = np.sum(x)
rs
#the sample standard dev
s = np.std(x, ddof=1)
s
#### **y** consist of the cube of **x** values
y = x**3
y
#### **m** is the mean of **y**
m = np.mean(y)
m
### Task 2
np.random.seed(rs)
s = np.random.binomial(20, 0.65, 150)
s
#plot the histogram
import matplotlib.pyplot as plt
plt.hist(s, bins = 12, color = 'brown', edgecolor='black')
plt.xlabel('values')
plt.ylabel('Frequency')
plt.title('Histogram of Sample...showing Frequency vs Value of S')
plt.show()
from scipy.stats import binom
n = 15
p = 0.35
p = binom.cdf(6, n, p) - binom.cdf(3, n, p)
p
### Task 3
import pandas as pd
df = pd.read_csv("prices.csv")
df.head(8)
x = np.array(df['A'])
x
y = np.array(df['B'])
y
from scipy.stats import linregress
lr = linregress(x,y)
lr_lst = [lr.intercept, lr.slope, lr.rvalue]
lr_lst
b0 = lr_list[0]
b1 = lr_list[1]
r = lr_list[2]
[b0, b1, r]
##### _plotting the graph..._
import matplotlib.pyplot as plt 
plt.scatter(x, y, color = 'green')
xvalues = np.linspace(min(x), max(x), 150)
yvalues = lr.intercept + lr.slope * xvalues
plt.plot(xvalues, yvalues, color = 'b')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.show()
### Task 4
cat_df = pd.read_csv("categories.csv")
cat_df.head()
cx = np.array(cat_df['Category X'])
cx
##### _Given the parameters..._
alpha = 0.01
µ = 31
##### _...using the previously imported `scipy.stats` package, import `ttest_1samp`_
from scipy.stats import ttest_1samp
ttest = ttest_1samp(cx, 31)
[ttest.statistic, ttest.pvalue]
#### Since the `p_value = 0.0075` is less than `alpha = 0.01`; We **reject** the null hypothesis.
### Task 5
##### _...using the previously imported `scipy.stats` package, import `norm`_
import numpy as np
from scipy.stats import expon
x = expon.rvs(size = 150, scale = 1/0.3)
xbar = np.mean(x)
xbar
##### _Using the Central Limit Theorem..._
mu = 1/0.3
std = 1/(0.3 * np.sqrt(150))

from scipy.stats import norm
z = norm.cdf(5.3, mu, std) - norm.cdf(3.5, mu, std)
z
#### Task 6
from scipy.stats import norm,fit
df = pd.read_csv('groups.csv')
