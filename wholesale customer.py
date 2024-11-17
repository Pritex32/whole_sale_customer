#!/usr/bin/env python
# coding: utf-8

# # Analysis by Prisca

# In[ ]:





# ## Description: Data about wholesale customer purchasing behavior.
# ## Features: Annual spending on various products (e.g., milk, grocery, frozen foods).
# ## The data set Multivariate refers to clients of a wholesale distributor. It includes the annual spending in monetary units (m.u.) on diverse product categories
# ## Dataset Source: Available on the UCI Machine Learning Repository.
# ## i will be using clustering and self organizing map to analyze this data

# In[ ]:





# ## columns explainations:
# ## Region Represents the geographic region of the customer.
# ### 1: Lisbon.
# ### 2: Oporto.
# ### 3: Other regions.
# ## Channel Represents the type of customer.
# ### 1: Hotel/Restaurant/Cafe (HoReCa).
# ### 2: Retail (e.g., shops or individual customers
# ## Fresh:The annual spending (in monetary units) on fresh products, such as vegetables, fruits, meat, and seafood,
# ## Milk:The annual spending (in monetary units) on milk and dairy products.
# ## Grocery: The annual spending (in monetary units) on grocery items, including non-perishables like canned goods or pantry staples.
# ## Frozen:The annual spending (in monetary units) on frozen foods, such as ice cream, frozen vegetables, and frozen meals.
# ## Detergents_Paper:The annual spending (in monetary units) on cleaning products and paper products, such as detergents, tissues, and paper towels.
# ## Delicassen:The annual spending (in monetary units) on delicatessen items, such as prepared foods, specialty products, or high-end foods often found in deli sections.
# 

# In[ ]:





# ## the aim of this analysis
# ### 1. Identify customer segments (e.g., bulk buyers vs. low spenders).
# ### 2. Optimize marketing efforts.
# ### 3. Tailor product offerings to specific customer needs.
# 

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # import libraries
import warnings
warnings .filterwarnings('ignore')


# In[49]:


df=pd.read_csv(r'C:\Users\USER\Documents\dataset\Wholesale customers data.csv') #load Dataset


# In[50]:


df.head()


# In[160]:


df.shape # size of the dataset


# In[55]:


df.duplicated().sum() # no duplicate values


# # Exploratory analysis

# In[161]:


df_grouping=df.groupby('Channel').mean() # overall filetring 


# In[57]:


df_grouping


# # High spending vs Low spending

# In[58]:


df_1=df[df['Channel']==1].sum()
plt.figure(figsize=(10,10))
sns.barplot(df_1,ci=None,palette='RdBu')
plt.title('purchasing behaviour of hotel/resturant') # hotels are the low spenders


# In[ ]:





# In[59]:


df_2=df[df['Channel']==2].sum()
plt.figure(figsize=(10,10))
sns.barplot(df_2,palette='Spectral')
plt.title('purchasing behaviour of retail customers')  # retails are the high spenders


# ## findings:
# ## 1.the both customer base have very low spending  on item Delicassen.
# ## 2. retailers/individuals are the customers with high puchasing power, that is they spend more .
# ## 3. hotel/resturant purchase frozen foods in a very high quantity.
# 
# ## recommendation:
# ## 1 .the marketing team can niche or narrow their focus on products that the individuals customer barely purchase such includes frozen and Delicassen products, they marketing team can include bonus to these product or implement discounts.
# ## 2.the hotels purchases only frozen foods in high quantity and lowest of Delicassen, the market should put effort to communicate benefits of other products to the hotels by emailing or adverts.
# ## 3.they should also figure out the position of theses products in the minds of the customers by gving out questionaires, this way they can know what and how to communicate the customers.
# 
# ## conclusion:
# ## the individuals customer base spend alot in buying products  

# In[ ]:





# In[ ]:





# ## Regional customer base

# In[162]:


region_1=df[df['Region']==1].sum()
plt.figure(figsize=(10,10))
sns.barplot(region_1,palette='viridis')
plt.title('lisbon region purchasing behaviour')


# In[ ]:





# In[62]:


region_2=df[df['Region']==2].sum()
plt.figure(figsize=(10,10))
sns.barplot(region_2, palette='magma')
plt.title('Oporto region purchasing behaviour')


# In[ ]:





# In[64]:


region_3=df[df['Region']==3].sum()
plt.figure(figsize=(10,10))
sns.barplot(region_3,palette='inferno')
plt.title('other region purchasing behaviour')


# In[ ]:





# # findings:
# ## All regions puchases fresh food and grocery in high quantity but very little of Delicassen items.
# ## products like frozen foods, detergent and delicassen are in very low quantity accross all regions.
# 
# # Recommendations:
# ## i suggest that the marketing team to focus their advert on these products with low purchases especially Delicassen.
# ## communicates these products to customers to emails and advert.
# ## they can promote the products by implementing discounts, bonuses and many packages.
# 
# # Conclusion
# ## All regions purchases fresh foods,milk and grocery in high quantity and the rest in very low quantity.

# In[ ]:





# In[ ]:





# # clustering using self organizing map

# In[81]:


from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler


# In[152]:


x=df[['Region','Fresh', 'Milk', 'Grocery', 'Frozen',
       'Detergents_Paper', 'Delicassen']]
y=df['Channel'] # feature selection


# In[83]:


scaler=MinMaxScaler(feature_range=(0,1))


# In[153]:


x_scaled=scaler.fit_transform(x) #scaling the features


# In[154]:


som=MiniSom(x=15,y=15,input_len=7,sigma=1.0,learning_rate=0.5) #initializing the model


# In[86]:


# initialize the weights
som.random_weights_init(x_scaled)


# In[87]:


# train the model
som.train_random(data=x_scaled,num_iteration=100)


# In[88]:


# visualization
from pylab import bone,pcolor,plot,colorbar


# In[89]:


plt.figure(figsize=(10,10))
bone()
pcolor(som.distance_map().T)
colorbar()
markers=['o','d']
colors=['g','r']
for i, xi in enumerate(x_scaled):
    w=som.winner(xi)
    plot( w[0] +0.5, w[1] +0.5, markers[y[i] % len(markers)],markeredgecolor=colors[y[i] % len(colors)], markersize=15,markerfacecolor=None)


# ## marker 'o' represent hotel which stands for 1, and 'd' represent retailers which stands for 2 and this project is to determine the high spenders and on what they spend on.
# ## from this map, it shows that marker 'D' that is diamond shaped, that is the white box are the obvious spenders.

# In[ ]:





# In[139]:


mappings=som.win_map(x_scaled) # getting the values inside the cluster D


# In[93]:


len(mappings)


# In[155]:


# it is grouped into 2 clusters
spenders1=mappings[(13,13)] 
spenders2=mappings[(5,12)]


# In[123]:


spender_df=np.concatenate([spenders1,spenders2],axis=0) # combining the mapping together


# In[157]:


spender_inv=scaler.inverse_transform(spender_df) # reversing the scaling


# In[156]:


data=pd.DataFrame(spender_inv) # converting the clusters to df


# In[130]:


data


# # Cluster interpretation

# In[141]:


spender1_inv=scaler.inverse_transform(spenders1)
spender2_inv=scaler.inverse_transform(spenders2)


# In[151]:


sns.barplot(spender1_inv)
plt.title('other region spending behaviour') #visualizing first cluster


# In[ ]:





# In[158]:


sp=[36847.0,	43950.0,	20170.0,	36534.0,	239.0,	47943.0]
plt.pie(sp,labels=['Fresh', 'Milk', 'Grocery', 'Frozen',
       'Detergents_Paper', 'Delicassen'],autopct='%1.1f%%')
plt.title('other region spending behaviour') # getting the % spent on each products


# ## findings:
# ## this region called 'other region', spend's very less on detergent and groceries that is 10.9% on grocery and no or 0.1% on detergent paper.

# In[ ]:





# In[150]:


sns.barplot(spender2_inv)
plt.title('Oporto spending behaviour') # visualizing second cluster


# In[ ]:





# In[159]:


sp2=[32717.0,	16784.0,	13626.0,	60869.0	,1272.0,	5609.03]
plt.pie(sp2,autopct='%1.1f%%',labels=['Fresh', 'Milk', 'Grocery', 'Frozen',
       'Detergents_Paper', 'Delicassen'])
plt.title('Oporto spending behaviour') # getting the % of each second cluster


# ## findings: 
# ## this clusters are likely hotels, the spend 25% on fresh foods, 46% on frozen food and little or 1.0% on Detergent paper abd 4.3% on Delicassen products. The spend less on other products outside frozen foods especiall from this region Oporto.
# 

# 

# # Findings:
# ## 1. Both clusters spends very low on Detergent papers.
# ## 2. there are 3 regions but only Oporto and other regions spends more.
# # Recommendations:
# ## 1.i suggest the marketing team to identify why customers don't purchase other products may be through questionaire and suggestion box.
# ## 2. the marketers should focus more to advertise these low purchase products such as detergent paper,delicassen especially to hotels/resturants in oporto and lisbon regions.
# ## 3. communicate product benefits to customers through emails.
# ## 4. include bonuses and incentives on those products.

# In[ ]:




