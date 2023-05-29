#!/usr/bin/env python
# coding: utf-8

# In[28]:


#numpy libraries
get_ipython().system('pip install numpy')


# In[29]:


import numpy as np


# In[30]:


a=5
print(a)
print(type(a))


# In[31]:


#defining 1D array
import numpy as np
a=np.array([1,2,3,4,5])
print(a)
print(type(a))
print(a.dtype)#data type


# In[32]:


b=np.array([1,2,3.5,4,5])
print(b)
print(type(b))#type show arrays
print(b.dtype)
print(b.dtype)


# In[33]:


#defining arrays 
a=np.arange(10)
print(a)
b=np.arange(20,30)
print(b)
c=np.arange(20,30,3)
print(c)
print(type(a),type(b),type(c))


# In[34]:


print(b)
print(b[0])
print(b[5])
print(b[-1])


# In[35]:


#slicing
print(b[:5])
print(b[5:])
print(b[4:7])
print(b[::2])
print(b[::-1])
print(b[5::-1])
print(b[0::-1])


# In[36]:


x=np.array((1,2,3,4,5))
print(x)
print(type(x))


# In[37]:


print(x.ndim)#to check dimension of the array
print(sum(x))
print(len(x))
lt=[1,2,3,4]
print(sum(lt))


# In[39]:


#defining 2d array
ar=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(ar)
print(type(ar))
print(ar.ndim)


# In[40]:


#indexing of 2d arrays
print(ar[2][1])
print(ar[0,2])


# In[27]:


#slicing in 2d arrays
ar=np.array([[1,2,3],[4,5,6],[7,8,9]])
r1=ar[:2,:2]
print(r1)
r2=ar[::-1,::-1]
print(r2)
r3=ar[::-1,:-3]
print(r3)
print(ar[:,0])
print(ar[0,:])
print(ar[(0,1,2),(0,1,2)])


# In[42]:


a=np.array([1,2,3,4,5])
b=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
print(b)
print(a.size)
print(b.size)#size show number of element
print(a.shape)
print(b.shape)#google it(rowsxcolumn)order of matrix


# In[43]:


c=np.array([[[1,2],[3,4],[5,6],[7,8]]])
print(c)
print(c.ndim)
print(c.shape)
print(c.size)
print(c.itemsize)#data size in byte or items size in bytes


# In[44]:


print(a.itemsize)
print(b.itemsize)


# In[ ]:





# In[49]:


print(b.dtype)
b=np.array([1,2,3,4,5],dtype='int16')#compressing the size of array
print(b.dtype)


# In[50]:


print(c.dtype)
c=np.array([1,2,3,4,5],dtype='int16')#compressing the size of array
print(c.dtype)


# In[54]:


#creating an array of zeros
a=np.zeros((2,4),dtype='int16')
print(a)
print(a.dtype)
print(a.itemsize)


# In[56]:


#creating an array of ones
a=np.ones((2,4),dtype='float16')
print(a)
print(a.dtype)
print(a.itemsize)


# In[57]:


a=np.ones((2,4),dtype='int16')
print(a)
print(a.dtype)
print(a.itemsize)


# In[59]:


a=np.ones((2,4))
print(a)
print(a.dtype)
print(a.itemsize)


# In[60]:


#creating an array having ALL SAME i.e.@44
a=np.full((3,4),44)
print(a)


# In[61]:


a=np.full((3,3),np.nan)
print(a)


# In[62]:


#creating a diagonal matrix
c=np.diag([10,20,30,40,50])
print(c)


# In[63]:


#creating an identity matrix
d=np.eye(5,dtype='int16')
print(d)
print(d.dtype)


# In[66]:


#changing the type of arrays
a=np.array([1,2,3,4,5])
print(a)
print(a.dtype)
b=a.astype(float)
print(b)
print(b.dtype)


# In[70]:


#reshaping the arrays
c=np.arange(1,13)
print(c)
x=c.reshape((3,4))
print(x)
y=c.reshape((2,3,2))#google it
print(y)
z=c.reshape((1,12))
print(z)
z1=c.reshape((12))
print(z1)


# In[73]:


import numpy as np
ar=np.array([[1,2,3],[4,5,6]])
print(ar)
#flatten and ravel methods
a=ar.ravel()
print(a)
print(ar)
b=ar.flatten()
print(b)


# In[75]:


#repeat:it repeats the element of an existing array
x=np.array([[10,20,30]])
print(x)
r=np.repeat(x,10,axis=0)#row wise
print(r)
s=np.repeat(x,10,axis=1)#column wise
print(s)


# In[85]:


#creating empty matrics or garbage value
e=np.empty((3,3),dtype='int16')#try it with
print(e)


# In[84]:


#random numbers
d=np.random.rand(3,4)
print(d)


# In[94]:


#random.randint()is used to create matrice>>>it returns random integer values
c=np.random.randint(-4,10,size=(2,3))
print(c)


# In[95]:


#view method:copy and view method and use copy in place of view
a=np.array([10,20,30,40,50])
print(a)
b=a.view()
print(b)
b[1]=200
print(b)
print(a)


# In[96]:


#trace:it will give sum all diagonal elements
print(np.trace(a))
print(np.trace(b))


# In[98]:


x=np.arange(11,23).reshape(3,4)
print(x)
y=x.ravel()#actual array is modified
z=x.flatten()#actual array is not modified
print(y)
print(z)
y[1]=100
z[1]=200
print(y)
print(z)
print(x)


# In[100]:


#MATHEMATICAL OPERATIONS
a=np.arange(1,9)
print(a)
print(a+2)
print(a*2)
print(a/2)
print(a//2)


# In[101]:


a=a.reshape(2,4)
print(a)
print(a)
print(a+2)
print(a*2)
print(a/2)
print(a//2)


# In[108]:


print(np.sin(90) )
print(a.transpose())
print(a.T())


# In[112]:


ar1=np.array([[1,2],[3,4]])
#trace()=sum of diagonal element
print(ar1.trace())


# In[114]:


#multiplication of two matrix
ar2=np.array([[1,2],[3,4]])
ar1=np.array([[1,2],[3,4]])
print(ar1.dot(ar2))
print(ar1@ar2)
print(ar1*ar2)#element multiplication
print(np.matmul(ar1,ar2))


# In[115]:


#determinant
print(np.linalg.det(ar1))


# In[119]:


#inverse of matrix
ar1=np.array([[1,2],[3,4]])
print(np.linalg.inv(ar1))


# In[ ]:




