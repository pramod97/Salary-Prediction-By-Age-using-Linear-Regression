import numpy                          # for arrays
import matplotlib.pyplot as plt       # for 
import seaborn as sns                 # for graph
import random                         # for random data

# create a function

def studentReg(ages_train,net_worth_train):
    
    # import linear regression from sklearn library
    
    from sklearn.linear_model import LinearRegression
    
    # fit this model to training data by creating object and calling the model
    
    reg=LinearRegression().fit(ages_train,net_worth_train)
    
    return reg

# creating random no, seed is for generating same value again and again

numpy.random.seed(42)

# create empty list to store random no generated

ages=[]

# generate random no and append those no in empty list

for li in range(250):
    ages.append(random.randint(18,75))   # randint generates random int between 18 n 75
    
# generate net_worth data 

net_worth=[li * 6.25 + numpy.random.normal(scale=40) for li in ages]

# converting ages and networth in array and reshaping them in 2d array

ages= numpy.reshape(numpy.array(ages),(len(ages),1))

net_worth= numpy.reshape(numpy.array(net_worth),(len(net_worth),1))

# seperate this data in train and teat data set using inbuild function

from sklearn.model_selection import train_test_split

ages_train, ages_test, net_worth_train, net_worth_test = train_test_split(ages,net_worth)

# train our model by calling our function

reg1 = studentReg(ages_train, net_worth_train) 

# chesk slope and intercept of train model

print("coefficient",reg1.coef_)
print("intercept",reg1.intercept_)

# calculate efficiency of model using .score

print("Training data",reg1.score(ages_train,net_worth_train))
print("Test data",reg1.score(ages_test,net_worth_test))


plt.figure(figsize=(12,10))

# calling regplot to plot data 

sns.regplot(x=ages_train, y=net_worth_train,scatter=True,color="b",marker="*")
 
# give label to x and y

plt.xlabel("Ages train")
plt.ylabel("Net worth trian")
plt.title("Regression plot")


plt.figure(figsize=(12,10))

# use scatterplot 

plt.scatter(ages_train,net_worth_train,color="b",label="train data")
plt.scatter(ages_test,net_worth_test,color="r",label="test data")

# predicting ages to get net worth

plt.plot(ages_test,reg1.predict(ages_test))

plt.xlabel("Ages")
plt.ylabel("Net worth")

# where to show label in graph 

plt.legend(loc=2)

# show graph

plt.show()

    
    
    
     




