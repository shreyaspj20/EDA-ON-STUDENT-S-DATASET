#!/usr/bin/env python
# coding: utf-8

# ### THIS IS THE EXPLORATORY DATA ANALYSIS PROJECT ON THE STUDENT'S PERFORMANCE DATASET.
# 

# #### 1.IMPORTING THE NECESSARY LIBRARIES OR PACKAGES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mat
import seaborn as sns


# #### 2.STORING THE DATA GIVEN TO US TO A DATAFRAME IN PANDAS

# In[2]:


df=pd.read_csv("StudentsPerformance.csv",sep=",")


# In[3]:


df.head()


# In[4]:


df.dtypes


# In[5]:


df.describe()


# In[6]:


import pandas_profiling as pp
pp.ProfileReport(df)


# From the above report ,we infer that there are no missing values in the given data and the data has 1000 observations in
# total.So lets understand some corellations using graphs and other visualizations

# In[7]:


sns.violinplot(y='math score',data=df)
#sns.violinplot(y='reading score',data=df)
#sns.violinplot(y='writing score',data=df)


# By analysing the above violinplots,we find that all the numerical data provided to us is well distributed and symmetric in nature.

# In[8]:


df[df['gender']=='female'].count()


# So there are 518 female and 482 males in our analysis dataset which is well balanced.

# In[9]:


df[(df['gender']=='female') & (df['math score']>80)].count()


# We see that only 68 females have scored above 80 in maths .
# 

# In[10]:


df[(df['gender']=='male') & (df['math score']>80)].count()


# We see that  108 males have scored above 80 in maths .This shows that males had a better understanding of maths than females.

# In[11]:


mat.scatter(df['gender'],df['math score'])
mat.xlabel("Gender")
mat.ylabel("math score")
mat.title("GENDER VS MATH SCORE")


# From this above scatter plot ,we find that the least score among males is around 25 wheraes there is a fair number of females in range of 0 to 40.This again proves that on an average,males were good at maths than females.

# In[12]:


correlation_df = df.corr()
correlation_df


# From the corellation score,we find that the reading and writing score are mostly dependent on each other with the corellation factor being 0.95. So a person who scores well in reading also scores well in writing test and vice versa is also true.

# Therefore,instead of having two seperate columns or reading score and writing score,we can by feature engineering create a new feature called read_write_score which holds the value of the average of a student's read and write score.

# In[13]:


df['read_write_score']=(df['reading score']+df['writing score'])/2


# In[14]:


df.drop(columns=["reading score","writing score"],inplace=True)


# So as said,we have created a new feature "read_write_score" and now we will compare the dependency of other features on the newly built feature.

# In[15]:


mat.scatter(df['gender'],df['read_write_score'])
mat.xlabel("Gender")
mat.ylabel("read_write_score")
mat.title("GENDER VS READ WRITE SCORE")
gro=df.groupby(df['gender'])
gro.describe()


# In[16]:


mat.scatter(df['race/ethnicity'],df['read_write_score'],edgecolors="black",marker=10)
mat.xlabel("Race group")
mat.ylabel("read_write_score")
mat.title("RACE VS READ WRITE SCORE")


# From the above scatterplot,we find that group D and group E have a consistent performance above score of 60 than the other groups.In group A,B and C ,we find maximum cases which are below 40 compared to the other groups.

# In[17]:


sns.jointplot(x="math score", y="read_write_score", data=df)


# In[18]:


import seaborn as sns
sns.distplot(df['math score'],kde=True)
sns.distplot(df['read_write_score'],kde=True)


# The above 2 visualizations show that maximum nuber of students score around 65 in maths as the graph peaks in that region .Similarly,maximum number of students score around 70 in both reading and writing tests.

# In[19]:


group=df.groupby(df['race/ethnicity'],axis=0)


# In[20]:


print(group.describe())
mean=group.agg(np.mean)


# In[21]:


mean


# In[22]:


mean[['math score','read_write_score']].plot(kind="bar",title="MEAN SCORES VS RACE")


# From the above bar plot ,we find that Group E performs better in both maths as well as reading and writing exams.Group E is followed by D,C,B,A.But there are much students in Group E who underperform as well when compared to Group D because the standard deviation of group E is greater than group D.
# Therefore group D has students who have consistency and dont score low.

# In[23]:


group1=df.groupby(df['parental level of education'],axis=0)


# In[24]:


group1.describe()


# It is very clear from the above table that the children whose parents are well qualified are scoring better in all the 3 subjects.The order of well qualified being(masters > bachelor's > associate's > college > some high school > high school).The marks scored also follows the same decreasing pattern.
# The main inference was that parents who were well qualified could help their children in learning than the less qualified parents,therefore scores of students are directly proportional to the parent's education.

# In[25]:


group2=df.groupby(df["lunch"],axis=0)


# In[26]:


X=list(group2.size())
mat.pie(X,labels=['free/reduced','standard'],shadow= True,
        explode=(0,0.1),
        autopct='%1.1f%%')


# This shows that that 355 students recieve free/reduced lunch and the remaining 645 recieve standard lunch.

# In[27]:


group2.describe()


# It can be found that students with reduced or free lunch ,students score less compared to students with standard lunch.It can be interpreted that students with lesser facilities score less in all the 3 subjects compared to people with facilities due to some reasons.

# In[28]:


mat.scatter(df['lunch'],df['math score'])
mat.title("Lunch VS Math score")
mat.xlabel("lunch")
mat.ylabel("math score")


# This plot above confirms the above made conclusion that students with standard lunch score better

# In[29]:


group3=df.groupby(df['test preparation course'],axis=0)


# In[30]:


group3['test preparation course'].size()


# This shows us that 358 students have taken and successfully completed the course and 642 have not shown interest in any such courses.

# In[31]:


mean1=group3.agg(np.mean)


# In[32]:


mean1.plot(kind="bar",title="Course VS Mean of scores")


# As expected , students who have completed the courses are more likely to score well in Maths as well as in reading and writing tests.But read_write_scores of both kind of students are almost same even though students who have taken the course have benefited only a little by the course. 

# In[33]:


df1=pd.get_dummies(df)
sns.heatmap(df1.corr())


# ## CONCLUSIONS.

# 1.Females perform well in reading and writing tests and males perform well in maths test.

# 2.It was found that group E and D proved superior than ther groups in order(E>D>C>B>A).

# 3.We infered that students whose parents hold a higher qualification perform better in all the exams.(masters>bachelor's>associate's>college>high school>some high school).

# 4.Students who recieved standard lunch performed better than free/reduced lunch provided students.

# 5.Students who took a test course and completed it successfully got bettre grades than the students who did not take them.

# In[ ]:





# # THANK YOU

# In[ ]:




