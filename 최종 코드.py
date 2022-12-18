#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action = 'ignore')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
 
get_ipython().system('apt -qq -y install fonts-nanum')
 
import matplotlib.font_manager as fm
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
mpl.font_manager._rebuild()
import time
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
plt.rc('font', family='Malgun Gothic')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')


# In[2]:


# Matplotlib 한글 폰트 오류 문제 해결
from matplotlib import font_manager, rc
font_path = 'c://Windows//Fonts//malgun.ttf' # 폰트 파일 위치
font_name = font_manager.FontProperties( fname = font_path ).get_name()
rc( 'font', family = font_name )
plt.rcParams['font.size'] = 15


# In[3]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[4]:


df1 = pd.read_csv('jeju.csv',encoding='euc-kr')


# In[5]:


df1.head()


# In[6]:


df1[df1['시도명']=='서귀포시'].count


# In[7]:


df1 = df1.sort_values(by=['년월'], ascending=True)
df1 = df1.reset_index(drop=True)
df1


# In[8]:


df1[(df1['년월']=='2019-12')]


# In[9]:


df1 = df1.loc[0:1483735,:]


# In[10]:


df1


# In[11]:


df1 = df1.drop(['데이터기준일자'],axis=1)


# In[12]:


mask = df1['시도명'].isin(['제주시'])


# In[13]:


df1[~mask].head()


# In[14]:


df1 = df1[~mask]


# In[15]:


df1


# In[16]:


#결측치 확인
df1.isnull().sum()


# In[17]:


df1 = df1.dropna(axis=0)


# In[18]:


# 데이터 타입 확인 및 변경

df1.dtypes


# In[19]:


# # 날짜 타입 변경 
# df1["년월"] = pd.to_datetime(df1["년월"],format='%Y-%m')
# df1.head()


# In[20]:


df1.rename(columns={'이용자 구분': '이용자구분'},inplace=True)


# ## 범부-연속-레이블 나누기

# In[154]:


X1 = df1[['년월','시도명','지역구분','읍면동명','업종코드','업종명','이용자구분','관광구분','성별','업종명 대분류']]
X2 = df1[['연령대','이용자수','이용건수','매장수','이용금액']]
y = df1[['이용금액']]


# In[76]:


X1_dum = pd.get_dummies(X1)


# In[77]:


X1_dum.head()


# In[78]:


X2.describe()


# In[79]:


pd.DataFrame(X2).hist(figsize=(20,10))


# In[155]:


# 위에 범주형은 처리했고 이제 연속형 처리

from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
scaler1.fit(X2)


# In[82]:


pd.DataFrame(X2).describe()


# In[83]:


#자료통합 및 저장하기

X2 = pd.DataFrame(X2)
X2.columns = ['연령대','이용자수','이용건수','매장수']
jeju2 = pd.concat([X1_dum,X2,y],axis=1)


# In[85]:


jeju2.tail()


# In[86]:


jeju2.to_csv('jeju2.csv',sep=',',encoding='euc-kr')


# In[87]:


df2 = pd.read_csv('jeju2.csv',encoding='euc-kr')


# In[88]:


df2


# In[38]:


df2.isnull().sum()


# In[39]:


df3 = df2.dropna(axis=0)


# In[40]:


df3


# ## Label encoding + 수치형만 따로 스케일링 시켜주기

# In[21]:


dtypes = df1.dtypes
encoders = {}
for column in df1.columns:
    if str(dtypes[column]) == 'object':
        encoder = LabelEncoder()
        encoder.fit(df1[column])
        encoders[column] = encoder

for column in encoders.keys():
    encoder = encoders[column]
    df1[column] = encoder.transform(df1[column])


# ### min-max scaling

# In[22]:


X2 = df1[['연령대','이용자수','이용건수','매장수']]
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
scaler1.fit(X2)

X2 = scaler1.transform(X2)
pd.DataFrame(X2).head()


# In[23]:


df1[['연령대','이용자수','이용건수','매장수']] = X2


# In[24]:


df1


# In[25]:


df3 = df1.dropna(axis=0)


# In[26]:


df3.info()


# ### Feature importance

# In[27]:


X_data = df3[df3.drop(['이용금액'],axis=1).columns]
y_data = df3[['이용금액']]


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3,shuffle=True)


# In[29]:


forest = RandomForestRegressor(n_estimators=100)
forest.fit(X_train, y_train)


# In[ ]:


#print(list(forest.feature_importances_))


# In[124]:


a =list(forest.feature_importances_)


# In[166]:


a


# In[125]:


importance = a
feature = X_train.columns
importances = pd.DataFrame()
importances['feature'] = feature
importances['importances'] = importance
importances['importances'] = importance
importances.sort_values('importances', ascending=False, inplace=True)
importances.reset_index(drop=True, inplace=True)
importances


# In[126]:


plt.figure(figsize=(10, 8))
sns.barplot(x='importances', y='feature', data=importances)
plt.title('Random Forest Feature Importances', fontsize=18)
plt.show()


# In[128]:


a=df3.corr()


# In[129]:


plt.figure(figsize=(10,10))
sns.heatmap(a)


# In[130]:


a


# ### 학습

# In[30]:


df3 = df3.drop(['이용건수','시도명','업종코드',],axis=1)
# df3 = df3.drop(['년월','업종명 대분류',''])
df3 = df3.drop(['관광구분','성별','지역구분','이용자구분','업종명 대분류','년월'],axis=1)


# In[31]:


df3.info()


# In[32]:


df3.to_csv('모델링용2.csv',encoding = 'utf-8')
df3 = pd.read_csv('모델링용2.csv') 


# ## train-test 나누기

# In[33]:


X = df3


# In[34]:


X = X.drop(['Unnamed: 0'],axis=1)
# X = X.drop(['Unnamed: 0.1'],axis=1)


# In[35]:


X.head()


# In[36]:


df4 = X.copy()


# In[37]:


df4.describe()


# In[38]:


df4_stats = df4.describe()
df4_stats = df4_stats.transpose()
df4_stats

def normalization( x ):
  return ( x - df4_stats[ 'mean' ] ) / df4_stats[ 'std' ]


# In[39]:


df4 = normalization(df4)
X['이용금액'] = df4['이용금액']
X


# In[40]:


X.info()


# In[41]:


train_df = X.sample( frac = 0.8, random_state = 42 )
test_df = X.drop( train_df.index )


# In[42]:


y_train = train_df.pop( '이용금액' )
y_test = test_df.pop( '이용금액' )


# In[43]:


X_train = np.asarray( train_df)
X_test = np.asarray( test_df )


# In[44]:


X_train = np.array( X_train )
X_test = np.array( X_test )


# In[45]:


y_train = np.array( y_train )
y_test = np.array( y_test )


# ## Random Forest

# In[53]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


# In[173]:


validation_model = RandomForestRegressor(n_jobs=-1, random_state=42)
cv = KFold(5, shuffle=True, random_state=42)
r2_scores = cross_val_score(validation_model, X_train, y_train, cv=cv,scoring='r2')
mse_scores = cross_val_score(validation_model, X_train, y_train, cv=cv,scoring='neg_mean_squared_error')
mae_scores = cross_val_score(validation_model, X_train, y_train, cv=cv,scoring='neg_mean_absolute_error')



print("교차 검증 r2 점수: {}". format(r2_scores))
print("교차 검증 r2 평균 점수: {:.2f}".format(r2_scores.mean()))
print("교차 검증 mse 점수: {}". format(mse_scores))
print("교차 검증 mse 평균 점수: {:.2f}".format(mse_scores.mean()))
print("교차 검증 mae 점수: {}". format(mae_scores))
print("교차 검증 mae 평균 점수: {:.2f}".format(mae_scores.mean()))


# In[174]:


model1 = RandomForestRegressor(n_jobs = -1, random_state = 42)
model1.fit(X_train, y_train)


# In[175]:


predictions = [round(value) for value in model1.predict(X_test)]
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
print('explained_variance_score: {}'.format(explained_variance_score(y_test, model1.predict(X_test))))
print('mean_squared_errors: {}'.format(mean_squared_error(y_test, model1.predict(X_test))))
print('mean_absolute_errors: {}'.format(mean_absolute_error(y_test, model1.predict(X_test))))
print('r2_score: {}'.format(r2_score(y_test, model1.predict(X_test))))


# In[ ]:


#정규화를 시켜주었기 때문에 실제 평균인 250만원의 0.05


# In[176]:


model1.score(X_train,y_train)
model1.score(X_test,y_test)


# In[177]:


sns.scatterplot(y_test, model1.predict(X_test))
plt.xlabel('실제값', fontsize=15)
plt.ylabel('예측값', fontsize=15)


# ## Linear Regression

# ## 목적: 미리 다 알아보고 결국에 모델 선택

# In[178]:


validation_model2 = LinearRegression()
cv = KFold(5, shuffle=True, random_state=42)
r2_scores = cross_val_score(validation_model2, X_train, y_train, cv=cv,scoring='r2')
mse_scores = cross_val_score(validation_model2, X_train, y_train, cv=cv,scoring='neg_mean_squared_error')
mae_scores = cross_val_score(validation_model2, X_train, y_train, cv=cv,scoring='neg_mean_absolute_error')


print("교차 검증 r2 점수: {}". format(r2_scores))
print("교차 검증 r2 평균 점수: {:.2f}".format(r2_scores.mean()))
print("교차 검증 mse 점수: {}". format(mse_scores))
print("교차 검증 mse 평균 점수: {:.4f}".format(mse_scores.mean()))
print("교차 검증 mae 점수: {}". format(mae_scores))
print("교차 검증 mae 평균 점수: {:.4f}".format(mae_scores.mean()))


# In[46]:


model2 = LinearRegression()
model2.fit(X_train, y_train)


# In[51]:


sns.scatterplot(y_test, model2.predict(X_test))


# In[48]:


model2.score(X_train,y_train)
model2.score(X_test,y_test)


# In[49]:


predictions = [round(value) for value in model2.predict(X_test)]
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
print('explained_variance_score: {}'.format(explained_variance_score(y_test, model2.predict(X_test))))
print('mean_absolute_errors: {}'.format(mean_absolute_error(y_test, model2.predict(X_test))))
print('mean_squared_errors: {}'.format(mean_squared_error(y_test, model2.predict(X_test))))
print('r2_score: {}'.format(r2_score(y_test, model2.predict(X_test))))


# ## SVM ==> 너무 데이터개수가 많아서 오히려 적합하지 않음
# 

# In[ ]:


from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
estimator = LinearSVC()
n_estimators = 10
n_jobs = 1
model = BaggingClassifier(base_estimator=estimator,
                          n_estimators=n_estimators,
                          max_samples=1./n_estimators,
                          n_jobs=n_jobs)
model.fit(X_train,y_train)


# In[41]:


df3.info()


# ## nn

# In[54]:


from sklearn.neural_network import MLPRegressor
validation_model5 = MLPRegressor(hidden_layer_sizes=(50,200,50),
                       max_iter = 400,activation = 'relu',
                       solver = 'adam')

cv = KFold(5, shuffle=True, random_state=42)
r2_scores = cross_val_score(validation_model5, X_train, y_train, cv=cv,scoring='r2')
mse_scores = cross_val_score(validation_model5, X_train, y_train, cv=cv,scoring='neg_mean_squared_error')
mae_scores = cross_val_score(validation_model5, X_train, y_train, cv=cv,scoring='neg_mean_absolute_error')


print("교차 검증 r2 점수: {}". format(r2_scores))
print("교차 검증 r2 평균 점수: {:.2f}".format(r2_scores.mean()))
print("교차 검증 mse 점수: {}". format(mse_scores))
print("교차 검증 mse 평균 점수: {:.4f}".format(mse_scores.mean()))
print("교차 검증 mae 점수: {}". format(mae_scores))
print("교차 검증 mae 평균 점수: {:.4f}".format(mae_scores.mean()))


# In[91]:


model5 = MLPRegressor(hidden_layer_sizes=(50,200,50),
                       max_iter = 400,activation = 'relu',
                       solver = 'adam')

model5.fit(X_train, y_train)


# In[92]:


model5.score(X_train,y_train)
model5.score(X_test,y_test)


# In[93]:


sns.scatterplot(y_test, model5.predict(X_test))
plt.xlabel('실제값', fontsize=15)
plt.ylabel('예측값', fontsize=15)


# In[88]:





# In[94]:


predictions = [round(value) for value in model5.predict(X_test)]
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
print('explained_variance_score: {}'.format(explained_variance_score(y_test, model5.predict(X_test))))
print('mean_absolute_errors: {}'.format(mean_absolute_error(y_test, model5.predict(X_test))))
print('mean_squared_errors: {}'.format(mean_squared_error(y_test, model5.predict(X_test))))
print('r2_score: {}'.format(r2_score(y_test, model5.predict(X_test))))


# 

# ### LightGBM

# In[186]:


import lightgbm as lgb
from lightgbm import  LGBMRegressor
validation_model3 =  LGBMRegressor(random_state=42)
cv = KFold(5, shuffle=True, random_state=42)
r2_scores = cross_val_score(validation_model3, X_train, y_train, cv=cv,scoring='r2')
mse_scores = cross_val_score(validation_model3, X_train, y_train, cv=cv,scoring='neg_mean_squared_error')
mae_scores = cross_val_score(validation_model3, X_train, y_train, cv=cv,scoring='neg_mean_absolute_error')


print("교차 검증 r2 점수: {}". format(r2_scores))
print("교차 검증 r2 평균 점수: {:.2f}".format(r2_scores.mean()))
print("교차 검증 mse 점수: {}". format(mse_scores))
print("교차 검증 mse 평균 점수: {:.4f}".format(mse_scores.mean()))
print("교차 검증 mae 점수: {}". format(mae_scores))
print("교차 검증 mae 평균 점수: {:.4f}".format(mae_scores.mean()))


# In[194]:


lgb = LGBMRegressor(random_state=42)
lgb.fit(X_train, y_train)


# In[195]:


sns.scatterplot(y_test, lgb.predict(X_test))


# In[196]:


lgb.score(X_train,y_train)
lgb.score(X_test,y_test)


# In[190]:


predictions = [round(value) for value in lgb.predict(X_test)]
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
print('explained_variance_score: {}'.format(explained_variance_score(y_test, lgb.predict(X_test))))
print('mean_absolute_errors: {}'.format(mean_absolute_error(y_test, lgb.predict(X_test))))
print('mean_squared_errors: {}'.format(mean_squared_error(y_test, lgb.predict(X_test))))
print('r2_score: {}'.format(r2_score(y_test, lgb.predict(X_test))))


# ## K-Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
validation_model7 = KNeighborsRegressor()
cv = KFold(5, shuffle=True, random_state=42)
r2_scores = cross_val_score(validation_model7, X_train, y_train, cv=cv,scoring='r2')
mse_scores = cross_val_score(validation_model7, X_train, y_train, cv=cv,scoring='neg_mean_squared_error')
mae_scores = cross_val_score(validation_model7, X_train, y_train, cv=cv,scoring='neg_mean_absolute_error')


print("교차 검증 r2 점수: {}". format(r2_scores))
print("교차 검증 r2 평균 점수: {:.3f}".format(r2_scores.mean()))
print("교차 검증 mse 점수: {}". format(mse_scores))
print("교차 검증 mse 평균 점수: {:.3f}".format(mse_scores.mean()))
print("교차 검증 mae 점수: {}". format(mae_scores))
print("교차 검증 mae 평균 점수: {:.3f}".format(mae_scores.mean()))


# In[55]:


a = pd.DataFrame({'모델':["RandomForest", "LinearRegression", "LGBM",'NN',"ExtraTree","K-Neighbors"],
                             'R2':[0.960,0.500,0.956,0.882,0.958,0.962],
                             'MAE':[0.055,0.228,0.067,0.101,0.056,0.054],
                             'MSE':[0.036,0.500,0.044,0.097,0.042,0.038]})


# In[56]:


fig = plt.figure( figsize = ( 20, 10) ) 
ax1 = fig.add_subplot( 1, 1, 1 )
ax2 = fig.add_subplot( 1, 1, 1 )
ax3 = fig.add_subplot(1,1,1)


ax1.plot(a['모델'], a['R2'], color = 'r', linewidth=2,  marker='o', label = '이용자수(100명)' )
ax2.plot(a['모델'], a['MAE'], color = 'b', linewidth=2,  marker='o' , label = '이용금액(원)')
ax3.plot(a['모델'], a['MSE'], color = 'g', linewidth=2,  marker='o', label = '이용건수(100건)' )

plt.title('모델별 R2 /  MAE / MSE 비교 ', size = 25)
plt.xlabel('모댈')
plt.legend(loc = 'upper right', labels=('R2','MAE','MSE')) 
plt.show()


# ### 예측값 구해보기

# In[ ]:


# 예측 템플릿 생성
# 코로나 사회적 거리두기 2022년 4월 ==> 그러면 예측은 올해 12월 내년 1월,2월 정도로만 예측
columns = ['년월','읍면동명','업종명','이용금액']
#년월2 = [202001,202005,202006,202007,202008,202009,202010,202011,202012]
#월2 = [1,2,3,4,5,6,7,8,9,10,11,12]

#년월2 = df2['년월1'].unique()
년월2 = [72]
읍면동명2 = df2['읍면동명1'].unique()
업종명2 = df2['업종명1'].unique()
이용금액2 = [0]


temp = []

for 년월 in 년월2:
    for 읍면동명 in 읍면동명2:
        for 업종명 in 업종명2:
            for 이용금액 in 이용금액2:
                temp.append([년월,읍면동명, 업종명,이용금액])
                
temp = np.array(temp)
temp = pd.DataFrame(data=temp, columns=columns)


print(temp.shape)
temp.head()


# In[ ]:


pred = temp[['년월','읍면동명','업종명']] # 모델에 넣을 변수만 추출


# In[ ]:


# NumPy 배열로 변경
pred = np.asarray(pred)
pred = np.array(pred)


# In[ ]:


temp.이용금액 = list(model1.predict(pred).round(0).astype(int))


# In[ ]:


# temp.pop('년월')
# temp.pop('읍면동명')
# temp.pop('업종명')


# In[ ]:


temp


# In[ ]:


j = temp.copy()


# In[ ]:


#코로나때라 값이 떨아짐...?


# In[ ]:


#plt.scatter(temp['업종명'],temp['이용금액'])


# In[ ]:


#일단 또 다른 모델을 이용하여 다른 예측값을 구해보기로 한다.


# In[ ]:





# In[ ]:


model2 = LinearRegression()


# In[ ]:


model2.fit(X_train, y_train)


# In[ ]:


model2.predict(X_test)


# In[ ]:


# 예측값과 실제값을 산점도로 비교
plt.scatter(y_test, model2.predict(X_test), alpha=0.4)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()


# - 실제값과 예측값이 선형성을 띄고 있지 않다.

# In[ ]:


print(model2.score(X_test, y_test))


# - 정확도가 매우 낮음을 알 수 있다.
# - 지금까지는 종속변수를 정규화하지 않은 상태로 모델을 만들었는데, 종속변수를 정규화를 해서 모델을 만들어 보기로 한다.

# In[ ]:


df2 = pd.read_csv('모델링용2.csv')


# In[ ]:


df2


# In[ ]:


df2_stats = df2.describe()
df2_stats = df2_stats.transpose()
df2_stats


# In[ ]:


def normalization(x):
    return ( x - df2_stats['mean'] ) / df2_stats['std']


# In[ ]:


# 정규화 실시 
df3 = normalization(df2)
df2['이용금액'] = df3['이용금액']
df2


# In[ ]:


# train, test 데이터로 나누기
train_df2 = df2.sample( frac = 0.8, random_state = 0 )
test_df2 = df2.drop( train_df2.index )


# In[ ]:


# y데이터 분리
y_train = train_df2.pop('이용금액')
y_test = test_df2.pop('이용금액')


# In[ ]:


# X,y 데이터를 NumPy 배열로 변경
X_train = np.asarray(train_df2)
X_test = np.asarray(test_df2)


# In[ ]:


X_train = np.array(train_df2)
X_test = np.array(test_df2)


# In[ ]:


y_train = np.array(y_train)
y_test = np.array(y_test)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model1 = RandomForestRegressor(n_jobs=-1, random_state=42)
model1.fit(X_train, y_train)


# In[ ]:


sns.scatterplot(y_test, model1.predict(X_test))


# In[ ]:


predictions = [round(value) for value in model1.predict(X_test)]
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
print('explained_variance_score: {}'.format(explained_variance_score(y_test, model1.predict(X_test))))
print('mean_squared_errors: {}'.format(mean_squared_error(y_test, model1.predict(X_test))))
print('r2_score: {}'.format(r2_score(y_test, model1.predict(X_test))))


# In[ ]:


model1.score(X_train,y_train)
model1.score(X_test,y_test)


# In[ ]:


# 왜 갑자기 음수값이 나오지..? ==> 정규화해서..?? ==> 해결함


# In[ ]:


# 다른 대안인 DecisionTreeRegressor


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, train_test_split
model3 = DecisionTreeRegressor()
model3.fit(X_train, y_train)


# In[ ]:


sns.scatterplot(y_test, model3.predict(X_test))


# In[ ]:


predictions = [round(value) for value in model3.predict(X_test)]
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
print('explained_variance_score: {}'.format(explained_variance_score(y_test, model3.predict(X_test))))
print('mean_squared_errors: {}'.format(mean_squared_error(y_test, model3.predict(X_test))))
print('r2_score: {}'.format(r2_score(y_test, model3.predict(X_test))))


# In[ ]:


model3.score(X_train,y_train)
model3.score(X_test,y_test)


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
model4 = ExtraTreesRegressor(n_jobs = -1, random_state = 42)
model4.fit(X_train, y_train)


# In[ ]:


sns.scatterplot(y_test, model4.predict(X_test))


# In[ ]:


predictions = [round(value) for value in model4.predict(X_test)]
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
print('explained_variance_score: {}'.format(explained_variance_score(y_test, model4.predict(X_test))))
print('mean_squared_errors: {}'.format(mean_squared_error(y_test, model4.predict(X_test))))
print('r2_score: {}'.format(r2_score(y_test, model4.predict(X_test))))


# In[ ]:


model4.score(X_train,y_train)
model4.score(X_test,y_test)

