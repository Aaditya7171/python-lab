
print("hello")

#variable
msg = ("hello world")
print(msg)
print(type (msg))
print(id(a))
b=3
print(id(b))   #variable with same value will have same MA
a=3
c=15/2
print(c)
d=15//2       #takes floor value i.e.,7.5=7
print(d) 
e=-15/2
print(e)
f=-15//2     #takes ceil value i.e.,-7.5=-8
print(f)    

#conditional statements
#if, else
a=10
if a>0:
    print("number is greater than 0")
else:
    print("not")

#if, else and elif
a=0
if a>0:
    print("positive")
elif a==0:
    print("Zero")
else:
    print("negative")

#finding the greatest number among 3 numbers
a=input(); b=input(); c=input()
a=int (a); b=int (b); c=int (c)
if a>b and a>c:
    print (a)
elif b>a and b>c:
    print (b)
else:
    print(c)4

#lists
list_name = [1,2,3,4]
print(list_name)
list_name = [1,2,3.5,4.5, "Hello"]
print(list_name)

#List slicing
lst1 = ["hello", 1,2.0,[1,2,3]]
print(lst1[0])
print(lst1[-1])
print(lst1[:])
print(lst1[0:4:2])
lst1=[1,2,3,4,5,6,7,8,9]
print(lst1[1::2])
print(lst1[::-1])

#'reverse' method on lists
lst1.reverse()
print(lst1)

#'remove method on lists
lst1.remove(8)
print(lst1)

#'delete method on lists
del lst1[3]
print(lst1)

#'discard' method on lists
lst1.pop(1)
print(lst1)
lst1.pop()
print(lst1)

#'append' method on lists
lst1=[1,2,3,4,5,6,7,8,9]
lst1.append(11)
print(lst1)
lst1.append("hello")

#'insert' method on lists
print(lst1)
lst1.insert()
print(lst1)

#'extend' method on lists
lst2=[14,15,16]
lst1.extend(lst2)
print(lst1)

#list sorting
lst3=[4,3,8,6,1,6,10,7]
lst4=sorted(lst3)
print(lst4)
print(lst3)
lst3.sort()
print(lst3)
lst3=[4,3,8,6,1,6,10,7]
lst4=sorted(lst3,reverse=True)
print(lst4)
lst1=[1,2,3,4]
#is is an identity operator
#in not in
print(3 in lst1)
print(3 not in lst1)
lst2=lst1
print(lst1 is lst2)

#loops and output formatting
#While loop
i=1
while i<=10:
    print(i)
    i+=1
else:
    print("out of while loop")

#for loop
for i in range(10,1,-2):
    print(i)
else:
    print("out of for loop")

#break & continue
lst1=[1,2,3,4,5,6,7,8,9]
for ele in range(len(lst1)):
    print(ele)
for i in range(1,12):
    if(i==4):
        break
    print(i)
else:
    print("out of loop")

#lsit comprehension
lst1=[for i in range(1,6)]
lst3=[[for i in range(1,4)]for j in range(1,4)]
print(lst3)

#tuples
#creation
t=() #empty tuple
t=(1,2,3)
print(t)
t=(1,2,3,4,"HEllo",(5,6,7),[1,2,3])

#tuple slicing 
print(t[0])
print(t[-1][0])
print(type(t[-1]))
print(type(t))
t[-1][0]=10;
print(t)
#tuple sorting
t1=(3,2,1)
t2=sorted(t1)
print(t2)
print(type(t2))
t3=tuple(t2)
l2=list(t3)
s="This,is,a,string"
l2=s.split(',')
print(l2)

#list split
s2="this is another string"
l2=s.split()
print(l2)
t4=(1,2,3)
min1=min(t4)
max1=max(t4)
sum1=sum(t4)
print(min1)
print(max1)
print(sum1)

#output formatting
print("minimum value is: ", min1, "maximum value is:", max1)

#sets
s=set()
print (type (s))
s={1,2,3,4,5,6,7,8,9,}
print(s)

#set functions
#set union
s1={1,2,3,4}
s2={5,6,7}
print(s1 | s2)
print(s1.union (s2))

#set intersection
s1={1,2,3,4,5,6}
s2={4,5,6,7}
print(s1.intersection (s2))
print(s1 & s2)

#set difference
s1={1,2,3,4,5}
s2={4,5,6}
print(s1 - s2)
print(s1.difference (s2))

#list of all methods of set
s=set()
print(dir (s))
#symmetric difference
s1={1,2,3,4,5}
s2={4,5,6}
print(s1 ^ s2)
print(s1.symmetric_difference(s2))


d={1:10,2:14,3:45}
print(d)
d[4]=16
print(d)
print(d.get(5))
print(d.get(4))

d={1:1,2:2,3:3,4:4}
d2={}
for k,v in d.items():
    d2[k]=v*v
print(d2)
s="Racecar"
k=0
if(s.lower()[::]==s.lower()[::-1]):
        print("given string is a palindrome")
else:
        print("given string is not a palindrome")

def ispalindrome(string):
    if(s.lower()[::]==s.lower()[::-1]):
        print("given string is a palindrome")
    else:
        print("given string is not a palindrome")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pip install pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Sample data (replace this with your actual data)
data = {
    'Semester': [1, 2, 3, 4, 5,6],
    'Subject': ['Math', 'Physics', 'Chemistry', 'Math', 'Physics', 'Chemistry'],
    'Year': [2019, 2019, 2020, 2020, 2021, 2021],
    'Grade': ['A', 'B', 'A', 'B', 'A', 'A']
}

df = pd.DataFrame(data)

# Convert categorical variables into numerical representations
df = pd.get_dummies(df, columns=['Subject', 'Grade'])

# Features and target variable
X = df.drop('Grade_A', axis=1)  # Assuming 'Grade_A' is the target variable
y = df['Grade_A']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Visualize results (replace this with your specific visualization)
plt.bar(range(len(predictions)), predictions, tick_label=X_test.index)
plt.xlabel('Sample Index')
plt.ylabel('Predicted Grade_A')
plt.title('Predictions for Each Sample')
plt.show()

