"""Baseline tabular demand forecast using a time-ordered holdout."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from .experiment import log_metrics
from .data_validation import validate_demand_frame

def make_demo_demand(n=180, seed=11):
    rng=np.random.default_rng(seed); dates=pd.date_range("2025-01-01",periods=n,freq="D")
    trend=np.linspace(80,125,n); weekly=12*np.sin(np.arange(n)*2*np.pi/7); noise=rng.normal(0,4,n)
    return pd.DataFrame({"date":dates,"demand":np.maximum(1,trend+weekly+noise)})

def train_forecast(frame=None):
    frame=frame.copy() if frame is not None else make_demo_demand()
    frame["date"]=pd.to_datetime(frame["date"]); validate_demand_frame(frame)
    frame=frame.sort_values("date").reset_index(drop=True)
    frame["day_of_week"]=frame.date.dt.dayofweek; frame["day_of_year"]=frame.date.dt.dayofyear
    frame["trend_index"]=np.arange(len(frame)); frame["lag_7"]=frame["demand"].shift(7).bfill()
    features=["day_of_week","day_of_year","trend_index","lag_7"]
    split=max(14,int(len(frame)*0.8)); split=min(split,len(frame)-1)
    model=RandomForestRegressor(n_estimators=100,random_state=42,min_samples_leaf=2)
    model.fit(frame.loc[:split-1,features],frame.loc[:split-1,"demand"])
    pred=model.predict(frame.loc[split:,features]); actual=frame.loc[split:,"demand"].to_numpy()
    mape=float(mean_absolute_percentage_error(actual,pred)); mae=float(mean_absolute_error(actual,pred))
    metrics={"mape":mape,"mae":mae,"train_rows":split,"test_rows":len(actual)}
    log_metrics("demand_forecast_baseline",metrics)
    return {"metrics":metrics,"predictions":pred.tolist(),"actual":actual.tolist()}
if __name__=="__main__": print(train_forecast()["metrics"])
