import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

covid_data=pd.read_csv(r"C:\Users\ANURAG\Desktop\Intellipat\Capstone Projects\COVID 19\Project-COVID-19-Analysis-1\covid_19_clean_complete.csv")
                       

covid_data

covid_data.shape

covid_data=covid_data.drop(columns=['Province/State'])

covid_data.head()

covid_data=covid_data.drop(columns=['Lat','Long'],axis=1)

covid_data.head()

covid_data['Date'] = pd.to_datetime(covid_data['Date'])

fig1 = px.line(global_trend, x='Date', y=['Confirmed', 'Recovered'], 
              title='COVID-19 Global Impact: Infections vs Recoveries',
              labels={'value': 'Total Cases', 'variable': 'Category'},
              color_discrete_map={'Confirmed': 'red', 'Recovered': 'green'})
fig1.show()

global_trend['Daily_Infections'] = global_trend['Confirmed'].diff().fillna(0)
global_trend['Daily_Recoveries'] = global_trend['Recovered'].diff().fillna(0)


fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=global_trend['Date'], y=global_trend['Active'], 
                         name='Confirmed', line=dict(color='orange')))
fig2.add_trace(go.Scatter(x=global_trend['Date'], y=global_trend['Recovered'], 
                         name='Daily New Recoveries', line=dict(color='blue')))
fig2.update_layout(title='Daily Rate of Change (Infection & Recovery)',
                  xaxis_title='Date', yaxis_title='New Cases Per Day')
fig2.show()

predict_df = global_trend[['Date', 'Confirmed']].rename(columns={'Date': 'ds', 'Confirmed': 'y'})

model = Prophet(interval_width=0.95, daily_seasonality=False)
model.fit(predict_df)



future = model.make_future_dataframe(periods=7)

forecast = model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7))

fig3 = plot_plotly(model, forecast)
fig3.update_layout(title='7-Day Forward Prediction of Total Cases')
fig3.show()




global_trend = covid_data.groupby('Date')[['Confirmed','Deaths','Active', 'Recovered']].sum().reset_index()

global_trend.head()
