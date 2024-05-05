#!/usr/bin/env python
# coding: utf-8

# ### EDA & Data Preprocessing on Google App Store Rating Dataset.

# ### 1. Import required libraries and read the dataset.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv(r"downloads/Apps_data.csv")


# ### Q2) Check the first few samples, shape, info of the data and try to familiarize yourself with different features.
# 

# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.head()


# In[6]:


data.tail()


# ### 3. Check summary statistics of the dataset. List out the columns that need to be worked upon for model building.

# In[7]:


data.describe()


# In[8]:


data.describe(include="all")


# The Features that are required to be worked upon for model building are : 
# 1. Rating
# 2. Type
# 3. Content Rating
# 4. Price
# 5. Category
# 6. Reviews

# ### 4. Check if there are any duplicate records in the dataset? if any drop them.

# In[9]:


duplicates = data[data.duplicated()]
duplicates


# In[10]:


data.drop_duplicates(inplace=True)


# In[11]:


data.shape


# ### Q5) Check the unique categories of the column 'Category', Is there any invalid category? If yes, drop them.

# In[12]:


data.Category.unique()


# In[13]:


invalid = data[data["Category"] == "1.9"]
invalid


# In[14]:


data.drop(10472, inplace=True)


# ### 6. Check if there are missing values present in the column Rating, If any? drop them and and create a new column as 'Rating_category' by converting ratings to high and low categories(>3.5 is high rest low)

# In[15]:


blank_rating = data[data["Rating"].isna()].index


# In[16]:


data.drop(blank_rating, inplace=True)


# In[17]:


def Rating_category(value):
    if value <= 3.5:
        return "Low"
    elif value > 3.5:
        return "High"


# In[18]:


data["Rating_category"] = data['Rating'].map(Rating_category)


# In[19]:


data.head()


# ### 7. Check the distribution of the newly created column 'Rating_category' and comment on the distribution.

# In[20]:


distribution = data["Rating_category"].value_counts()
distribution


# In[21]:


data["Rating_category"].hist()
plt.title("Distribution of Rating_category")


# From this information, we can see that the distribution of the 'Rating_category' column is imbalanced, with a significantly larger number of videos categorized as 'High' rating compared to 'Low' rating. This indicates that the majority of videos in the dataset have been classified as having a 'High' rating.
# 
# The histogram plot also confirms this observation, showing a skewed distribution towards the 'High' rating category, with a much smaller number of videos falling into the 'Low' rating category.

# ### 8. Convert the column "Reviews'' to numeric data type and check the presence of outliers in the column and handle the outliers using a transformation approach.(Hint: Use log transformation)

# In[22]:


data["Reviews"].dtypes


# In[23]:


data[data["Reviews"] == "3.0M"]


# In[24]:


data["Reviews"] = data["Reviews"].str.replace(".0M","000000")


# In[25]:


data["Reviews"] = data["Reviews"].astype(int)


# In[26]:


data.dtypes["Reviews"]


# In[27]:


data.Reviews.describe()


# In[28]:


sns.boxplot(x=data["Reviews"])


# In[29]:


log10 = np.log10(data["Reviews"])
log10.describe()


# In[30]:


sns.boxplot(x=log10, color="blue" , showmeans = True)
plt.title("BoxPlot for Analyzing Outlier's after Log transformation.")


# In[31]:


data["Reviews"] = log10


# In[32]:


data.head(5)


# ### 9. The column 'Size' contains alphanumeric values, treat the non numeric data and convert the column into suitable data type. (hint: Replace M with 1 million and K with 1 thousand, and drop the entries where size='Varies with device')

# In[33]:


data["Size"]


# In[34]:


data["Size"] = data["Size"].apply(lambda x : x.replace(",",""))


# In[35]:


data["Size"] = data["Size"].str.replace("M","000000")


# In[36]:


data["Size"] = data["Size"].str.replace("k","000")


# In[37]:


Varies_with_device = data[data["Size"] == "Varies with device"].index
Varies_with_device


# In[38]:


data.drop(Varies_with_device,inplace=True)


# In[39]:


data["Size"].convert_dtypes()


# ### 10. Check the column 'Installs', treat the unwanted characters and convert the column into a suitable data type.

# In[40]:


data["Installs"]


# In[41]:


data["Installs"] = data["Installs"].str.replace("+","").replace(",","")


# In[42]:


data["Installs"].convert_dtypes()


# In[43]:


data.head()


# ### 11. Check the column 'Price' , remove the unwanted characters and convert the column into a suitable data type.

# In[44]:


data["Price"]


# In[45]:


data["Price"].unique()


# In[46]:


data["Price"] = data["Price"].apply(lambda x : x.replace(",",""))


# In[47]:


data["Price"] = data["Price"].str.replace("$", "")


# In[48]:


data["Price"].convert_dtypes()


# ### 12. Drop the columns which you think redundant for the analysis.(suggestion: drop column 'rating', since we created a new feature from it (i.e. rating_category) and the columns 'App', 'Rating' ,'Genres','Last Updated', 'Current Ver','Android Ver' columns since which are redundant for our analysis)

# In[49]:


data.columns


# In[50]:


data.drop(["App","Rating","Genres","Last Updated","Current Ver","Android Ver"], axis = 1,inplace = True)


# In[51]:


data.head()


# ### 13. Encode the categorical columns.

# In[52]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


# In[53]:


data["Category"] = labelencoder.fit_transform(data["Category"])


# In[54]:


data.head()


# In[55]:


data["Content Rating"] = labelencoder.fit_transform(data["Content Rating"])


# In[56]:


data["Type"] = labelencoder.fit_transform(data["Type"])


# In[57]:


data["Rating_category"] = labelencoder.fit_transform(data["Rating_category"])


# In[58]:


data.head()


# ### 14. Segregate the target and independent features (Hint: Use Rating_category as the target)

# In[59]:


X = data.drop("Rating_category", axis=1)
Y = data[["Rating_category"]]


# ### 15. Split the dataset into train and test.

# In[60]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30 , random_state=1)


# ### 16. Standardize the data, so that the values are within a particular range.

# In[61]:


from sklearn.preprocessing import StandardScaler


# In[62]:


data['Installs'] = data['Installs'].apply(lambda x : x.replace(',','').replace('+','')).astype(int)
data['Size'] = data['Size'].apply(lambda x : x.replace(',','')).astype(float)
data['Price'] = data['Price'].apply(lambda x : x.replace(',','')).astype(float)


# In[63]:


scaler = StandardScaler()


# In[64]:


data = scaler.fit_transform(data)
data


# In[65]:


data = pd.DataFrame(data)


# In[75]:


data

