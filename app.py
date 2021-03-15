import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns

data = pd.read_csv('housing.csv')

st.write("""
# House Price Prediction App

This app predicts the **house price**!
""")
st.write("---")

prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

st.sidebar.header('Input Parameters')

def user_input_features():
    data = pd.read_csv('housing.csv')
    features = data.drop('MEDV', axis = 1)
    RM = st.sidebar.slider('RM', float(features.RM.min()), float(features.RM.max()), float(features.RM.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO', float(features.PTRATIO.min()), float(features.PTRATIO.max()), float(features.PTRATIO.mean()))
    LSTAT = st.sidebar.slider('LSTAT', float(features.LSTAT.min()), float(features.LSTAT.max()), float(features.LSTAT.mean()))
    data = {
            'RM': RM,
            'PTRATIO': PTRATIO,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.header('Data')
st.write(data.head(10))
st.write('---')

st.header('Summary of features of data')
st.write(''' 
- **RM**: This is the average number of rooms per dwelling
- **PTRATIO**: This is the pupil-teacher ratio by town
- **LSTAT**: This is the percentage lower status of the population
- **MEDV**: This is the median value of owner-occupied homes in $1000s
''')

st.header('Pairplot')
plot = sns.pairplot(data, size=4)
st.pyplot(plot, bbox_inches='tight')
st.write('We can spot a linear relationship between ‘RM’ and House prices ‘MEDV’. In addition, we can infer from the histogram that the ‘MEDV’ variable seems to be normally distributed but contain several outliers.')

st.header('Correaltion heatmap')
cm = np.corrcoef(data.values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size': 15},
                yticklabels=cols,
                xticklabels=cols)
st.pyplot(hm, bbox_inches='tight')

minimum_price = np.amin(prices)
maximum_price = np.amax(prices)
mean_price = np.mean(prices)
median_price = np.median(prices)
std_price = np.std(prices)

st.header('Statistics')
st.write('**Minimum Price**')
st.write(minimum_price)

st.write('**Maximum Price**')
st.write(maximum_price)

st.write('**Mean Price**')
st.write(mean_price)

st.write('**Median Price**')
st.write(median_price)

st.write('**Standard Deviation of prices**')
st.write(std_price)
st.write('---')

st.header('Selected Input parameters')
st.write(df)
st.write('---')

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state = 42)

model_dt = DecisionTreeRegressor(max_depth=4)
model_dt.fit(X_train, y_train)

# Apply Model to Make Prediction
prediction = model_dt.predict(df)

model = RandomForestRegressor()
model.fit(X_train, y_train)

predict = model.predict(df)

st.header('Prediction of MEDV (Random Forest)')
st.write(prediction)
st.write('---')

st.header('Prediction of MEDV (Decision Tree)')
st.write(predict)
st.write('---')

st.set_option('deprecation.showPyplotGlobalUse', False)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, data)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, data, plot_type="bar")
st.pyplot(bbox_inches='tight')