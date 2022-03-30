import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

boston = load_boston()
df = pd.DataFrame(boston.data,columns=boston.feature_names)
df['target']=boston.target
X=df.iloc[:,:-1]
y=df[['target']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
lm=LinearRegression()
lm.fit(X_train,y_train)

pickle.dump(lm, open('model.pkl','wb'))


model=pickle.load(open('model.pkl','rb'))

