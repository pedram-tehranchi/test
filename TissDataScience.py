import pandas as pd
import numpy as np
import holidays
from datetime import datetime, timedelta
from ptrack import provider
import pytz
import requests
import io
import warnings
from sklearn.preprocessing import StandardScaler
import joblib
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import portion as P

warnings.filterwarnings('ignore')

provider.set_url("grpc.edgecom.io")


class FeatureEngineering:
    def __init__(self):
        pass
    
    def ontario(self, dataset,daily=False):
        # adding holidays to dataframe
        dataset = self.to_datetime(dataset, 'time')
        if daily is False:
            dataset = self.TimeSeries().freq_data(dataset)
        
        dataset = self.add_holidays_to_dataframe(dataset, "time", "CA", ["ON"])
        # select features
        dataset = dataset[['time', 'temperature',
                                'Ontario Demand','is_holiday', 'ieso_demands']]
        # adding wind_chill feature
#         dataset = self.add_wind_chill(dataset)
        # adding time features

        dataset = self.add_time_feature(dataset,daily)


        # adding lags
        dataset.sort_index(inplace=True)

        dataset = self.add_demands_lag(dataset,daily)

        return dataset
        
    def to_datetime(self, df, column_name):
        df[column_name] = pd.to_datetime(df[column_name])
        return df
        
    def add_wind_chill(self, df):
        """`temperature` should be in °C and `wind_speed` should be in km/h"""
        df['wind_chill'] = (
            13.12 + 0.6215 * df.temperature - 11.37 * df.wind_speed ** .16
            + 0.3965 * df.temperature * df.wind_speed ** .16)
        return df

    def impute_missing_values(self, df):
        df.fillna(method='bfill', inplace=True)

    def add_demands_lag(self, df,daily=False):
        if daily is False:
            lags = [24, 25, 26, 27, 46, 47, 48, 49, 50, 70, 71, 72, 73, 74, 94, 95, 96, 97, 98]
        else:
            lags = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24, 25, 26, 27, 46, 47, 48, 49]
            
        for i in lags:
            df['Ontario_Demand_{}'.format(i)] =df['Ontario Demand'].shift(i)
        
        if daily is False:
            lags_ieso = [-5, -4, -3, -2, -1, 2,3, 23, 1,22, 24, 21, 25, 47, 48, 46, 71,95, 72, 70, 96, 20, 94, 45, 49, 26, 73, 69, 97]
        else:
            lags_ieso = [-5, -4, -3, -2, -1,1, 2,3,4 ,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
                         22, 23, 24]
            
        for i in lags_ieso:
            df['ieso_Demand_{}'.format(i)] = df['ieso_demands'].shift(i)
        return df

    def add_time_feature(self, data,daily=False):
        if daily is False:
            data['hour'] = data['time'].dt.hour
        data['dayofweek'] = data['time'].dt.dayofweek
        data['quarter'] = data['time'].dt.quarter
        data['month'] = data['time'].dt.month
        data['year'] = data['time'].dt.year
        data['weekofyear'] = data['time'].dt.weekofyear
        data['WEEKDAY'] = np.where(data['time'].dt.dayofweek < 5, 0, 1)
        return data

    def add_holidays_to_dataframe(self,
        dataframe, date_field, country_code, province: list = None):
        time = dataframe[date_field]
        country_holidays = holidays.CountryHoliday(country_code)
        country_holidays.prov = province
        cols = list(dataframe.columns)
        cols.append("StateHoliday")
        dataframe["StateHoliday"] = 0
        if dataframe[date_field].dtypes == "datetime64[ns]":
            dataframe[date_field] = dataframe[date_field].dt.strftime("%Y-%m-%d")
        dataframe.set_index(date_field, inplace=True)
        for i in zip(dataframe.index):
            if country_holidays.get(i[0]):
                dataframe["StateHoliday"].loc[i[0]] = 1
        dataframe.reset_index(inplace=True)
        dataframe.columns = cols
        dataframe[date_field] = pd.to_datetime(dataframe[date_field])
        dataframe.rename(columns={'StateHoliday':'is_holiday'}, inplace=True)
        dataframe['time'] = time
        return dataframe   
    
    
    class TimeSeries:
        def __init__(self):
            pass
        
        def freq_data(self, dataframe, subset='time', keep='first',
                      method='bfill', freq='H'):
            dataframe = dataframe.drop_duplicates(subset=subset, keep=keep)
            dataframe.set_index(subset,inplace=True)
            dataframe = dataframe.asfreq(freq=freq,method=method)
            dataframe.reset_index(inplace=True)
            return dataframe
        
        
class Data:
    
    def __init__(self):
        self.start_hour = 240
        self.end_hour = 240
        
    def toronto_time(self):
        tz = pytz.timezone('Canada/Eastern')
        toronto_now = datetime.now(tz)
        return toronto_now

    def ieso_predict(self, start_hour=None, end_hour=None):
        if start_hour == None:
            start_hour = self.start_hour
            end_hour = self.end_hour
        temp_time = self.toronto_time()
        start = temp_time - timedelta(hours=start_hour)
        end = temp_time + timedelta(hours=end_hour)
        res = provider.ieso_projected(start, end)
        data = pd.DataFrame(columns=['time'])
        list_time = []
        list_demands = []
        for i in res:
            for j in range(0,len(i.demands)):
                list_time.append(i.date + timedelta(hours=j))
            for n in i.demands:
                list_demands.append(n)
        data = pd.DataFrame()
        data['time'] = list_time
        data['ieso_demands'] = list_demands
        data['ieso_demands'] = data['ieso_demands'].shift(1)
        return data

    def ontario_demand_history(self, start_hour=None):
        start_time = self.toronto_time()
        if start_hour == None:
            end_time = start_time - timedelta(hours=self.start_hour)
        else:
            end_time = start_time - timedelta(hours=start_hour)
        start = datetime(start_time.year,
                         start_time.month, start_time.day, start_time.hour)
        end = datetime(end_time.year,
                       end_time.month, end_time.day, end_time.hour)
        res = provider.actual_demand(end, start)  
        data = pd.DataFrame()
        list_time = []
        list_demand = []
        for i in res:
            list_time.append(i.date)
            list_demand.append((i.demand))
        data['time'] = list_time
        data['Ontario Demand'] = list_demand
        return data
    
    def weather_forecast_daily(self):
        url = "https://toronto.weatherstats.ca/download.html"
        payload={'formdata': 'ok',
        'type': 'forecast_daily',
        'limit': '12',
        'submit': 'Download'}
        files=[]
        headers = {}
        response = requests.request("POST", url, headers=headers,
                                    data=payload, files=files)
        urlData = response.content
        rawData = pd.read_csv(io.StringIO(urlData.decode('utf-8')))
        rawData['date_time_local'] = pd.to_datetime(rawData['date_time_local'])
        rawData = rawData[['date_time_local','period_string','temperatures']]
        rawData['temperatures'] = rawData['temperatures'].str.extract('(\d+)').astype(int)
        return rawData

    def weather_forecast(self):
        url = "https://toronto.weatherstats.ca/download.html"
        payload={'formdata': 'ok',
        'type': 'forecast_hourly',
        'limit': '50',
        'submit': 'Download'}
        files=[]
        headers = {}
        response = requests.request("POST", url, headers=headers,
                                    data=payload, files=files)
        urlData = response.content
        rawData = pd.read_csv(io.StringIO(urlData.decode('utf-8')))
        rawData = rawData[['period_string','temperature','wind_speed']]
        rawData = rawData.rename(columns={'period_string': 'time'})
        rawData['time'] = pd.to_datetime(rawData['time'])
        rawData.drop_duplicates(subset='time',inplace=True,ignore_index=True)
        start_time = self.toronto_time()
        start = datetime(start_time.year, start_time.month,
                         start_time.day, start_time.hour)
        end = start + timedelta(hours=25)
        rawData = rawData[(rawData.time > start) &
                          (rawData.time <= end)].reset_index(drop=True)
        last_row = rawData[-1:].copy()
        last_row['time'][-1:] = last_row['time'][-1:] +timedelta(hours=1)
        rawData = rawData.append(last_row, ignore_index=True)
        return rawData

    def weather_history(self, limit=1000):
        url = "https://toronto.weatherstats.ca/download.html"
        payload={'formdata': 'ok',
        'type': 'hourly',
        'limit': '{}'.format(limit),
        'submit': 'Download'}
        files=[]
        headers = {}
        response = requests.request("POST", url, headers=headers,
                                    data=payload, files=files)
        urlData = response.content
        rawData = pd.read_csv(io.StringIO(urlData.decode('utf-8')))
        rawData = rawData[['date_time_local','temperature']]
        return rawData

    def merge_data(self):
        full_weather = self.weather_forecast()
        ieso_df = self.ieso_predict()
        od_df = self.ontario_demand_history()
        demand_joined = ieso_df.set_index('time').join(od_df.set_index('time')).reset_index()
        full_joined = demand_joined.set_index('time').join(full_weather.set_index('time')).reset_index()
        return full_joined
    
    def merged_history_data(self, limit = 220, start_hour=220, end_hour = 10):
        full_weather = self.weather_history(limit)
        full_weather.rename(columns={'date_time_local':'time'}, inplace=True)
        full_weather['time'] = pd.to_datetime(full_weather['time'])
        ieso_df = self.ieso_predict(start_hour, end_hour)
        od_df = self.ontario_demand_history(start_hour)
        demand_joined = ieso_df.set_index('time').join(od_df.set_index('time')).reset_index()
        full_joined = demand_joined.set_index('time').join(full_weather.set_index('time')).reset_index()
        full_joined = full_joined.dropna()
        return full_joined
    
    def convert_forecast_daily(self, data):
        nu = data.index[data['period_string'] == 'Tonight'].tolist()[0]
        df1 = data.iloc[:nu+1]
        df1_agg = df1.groupby(df1.index//2).agg({'date_time_local':'max','period_string':'min','temperatures':'mean'})
        df2 = data.iloc[nu+1:].reset_index(drop=True)
        df2_agg = df2.groupby(df2.index//2).agg({'date_time_local':'max','period_string':'min','temperatures':'mean'})
        concated_data = pd.concat([df1_agg, df2_agg]).reset_index(drop=True)
        list_dates = pd.date_range(start=concated_data.date_time_local[0],end=concated_data.date_time_local[0]+timedelta(days=len(concated_data)-1)).to_pydatetime().tolist()
        concated_data['date_time_local'] = list_dates
        concated_data['date_time_local'] = pd.to_datetime(concated_data['date_time_local']).dt.date
        return concated_data
    
    def daily_next_days(self):
        weather_daily = self.weather_forecast_daily()
        converted_daily = self.convert_forecast_daily(weather_daily)
        merged = self.merge_data()
        converted_daily = converted_daily.rename(columns={'date_time_local': 'time','temperatures': 'temperature'})
        converted_daily = converted_daily.drop('period_string',axis=1)
        data = merged[['time', 'Ontario Demand','ieso_demands']]
        max_daily = data.groupby(pd.Grouper(key='time', freq='D')).max()
        max_daily.reset_index(inplace=True)
        merged_data = pd.merge(converted_daily.set_index('time'),max_daily.set_index('time'), how='outer', left_index=True, right_index=True).reset_index()
        return merged_data


class Operation:
    
    def __init__(self):
        self.dt = Data()
        self.op = Ontario_pipeline()

    def daily_demand_prediction(self, merge_df = None):
        
        if merge_df is None:
            merge_df = self.dt.merge_data()
            result = self.op.predict(merge_df)
            start_time = self.dt.toronto_time()
            start = datetime(start_time.year, start_time.month, start_time.day, start_time.hour)
            end = start + timedelta(hours=24)
            result = result[(result.time >= 
                             start)&(result.time <= end)].reset_index(drop=True)[['time','Ontario Demand','demand_predicted', 'temperature','ieso_demands']]
        else:
            result = self.op.predict(merge_df)
            result = result[['time','Ontario Demand','demand_predicted',
                              'temperature','ieso_demands']]
        return result
    
    def _demand_current_year(self):
        end_time = self.dt.toronto_time()
        start = datetime(end_time.year, 1, 1, 0)
        end = datetime(end_time.year, end_time.month, end_time.day, end_time.hour)
        res = provider.actual_demand(start, end)
        data = pd.DataFrame()
        list_time = []
        list_demand = []
        for i in res:
            list_time.append(i.date)
            list_demand.append((i.demand))
        data['time'] = list_time
        data['Ontario Demand'] = list_demand
        return data
    
    def _max_daily(self, data):
        data = data[['time', 'Ontario Demand']]
        max_daily = data.groupby(pd.Grouper(key='time', freq='D')).max()
        max_daily.reset_index(inplace=True)
        return max_daily
    
    def _get_percent(self, data):
        percent = list()
        for i in range(len(data)):
            if 19794 < data['Ontario Demand'][i] < 21262.36:
                percent.append(65)
            elif 21262.36 < data['Ontario Demand'][i] < 21869.02:
                percent.append(75)
            elif 21869.02 < data['Ontario Demand'][i]:
                percent.append(85)
            else:
                percent.append((data['Ontario Demand'][i] / 19794) * 65)
        return percent
    
    def _get_percent_month(self, data):
        quantile_0 = [19128,18744,17059,14841,14577,16709,19314,18638,16458,16347,17881,18987]
        quantile_33 = [19641.04,19461,18032.91,16027.78,16035.21,18821.82,20901,20688.58,17661.89,16645.4,18569.42,19391.55]
        quantile_66 = [20107.68, 19792.0, 18458.04, 16873, 17130.74,20291.86,21579.0,21388.54,19735.18,16998.02,19006.82,19842.6]
        quantile_80 = [20373.6,19946.2,18876.4,17350.8,18112.4,20753,22469.2,22000.4,20436.2,17286.4,19163.4,20057]
        max_list = [21525, 20500, 20263, 18011, 20649, 22258, 24446, 23823, 23240, 18329, 20152, 20974]
        percent = list()
        for i in range(len(data)):
            if quantile_0[data['month'][i]-1] < data['Ontario Demand'][i] < quantile_33[data['month'][i]-1]:
                percent.append(65)
            elif quantile_33[data['month'][i]-1] < data['Ontario Demand'][i] < quantile_66[data['month'][i]-1]:
                percent.append(75)
            elif quantile_66[data['month'][i]-1] < data['Ontario Demand'][i]:
                percent.append(85)
            else:
                percent.append((data['Ontario Demand'][i]/quantile_0[data['month'][i]-1])*65)
        return percent

    def _prob_peak(self, data):
        data['prob'] = self._get_percent(data)
        flag = list()
        top_demands = [21963.3]
        data['percent'] = 0
        data['percent'] = (data['Ontario Demand'] / 25000) * 100
        for i in range(len(data)):
            if (data['Ontario Demand'][i] + 200 > sum(top_demands) / len(top_demands)) & (data.prob[i] == 85):
                top_demands.append(data['Ontario Demand'][i])
                if len(top_demands) > 10:
                    top_demands.sort(reverse=True)
                    top_demands = top_demands[:-1]
                flag.append(True)
            else:
                flag.append(False)
        data['flag'] = flag
        data.prob = np.where((data.flag == False) & (data.prob == 85), 80, data.prob)
        data.prob = np.where((data.flag == True) & (data.prob == 85), data.percent, data.prob)
        data.prob = np.where((data.prob < 80), data.percent - 10, data.prob)
        return data

    def predict_peak(self, merged_df = None,data_x=None,monthly=False):
        if merged_df is None:
            merged_df = self.dt.merge_data()
        demand_current = self._demand_current_year()
        daily = self._max_daily(demand_current)
        daily['month'] = pd.DatetimeIndex(daily['time']).month
        result = self.op.predict(merged_df)
        start_time = self.dt.toronto_time()
        start = datetime(start_time.year, start_time.month, start_time.day, 0)
        end = start + timedelta(hours=24)
        daily = daily[:-1]
        result = result[(result.time >= start) & (result.time <= end)].reset_index(drop=True)[['time', 'demand_predicted','Ontario Demand']]
        result.demand_predicted = np.where((result['Ontario Demand'] >0) , result['Ontario Demand'], result.demand_predicted)
        result = result[['time', 'demand_predicted']]
        max_forecast = result[result['demand_predicted'] == max(result['demand_predicted'])].rename(
            columns={'demand_predicted': 'Ontario Demand'})
        max_forecast['month'] = pd.DatetimeIndex(max_forecast['time']).month
        daily = pd.concat([daily, max_forecast]).reset_index(drop=True)
        
        if monthly is True:
            start_month = datetime(start_time.year, start_time.month, 1, 0)
            end_month = end
            quantile_80 = [20373.6,19946.2,18876.4,17350.8,18112.4,20753,22469.2,22000.4,20436.2,17286.4,19163.4,20057]
            max_list = [21525, 20500, 20263, 18011, 20649, 22258, 24446, 23823, 23240, 18329, 20152, 20974]
            monthly = daily[(daily.time >= start_month) & (daily.time <= end_month)].reset_index(drop=True)
            df_next = self.op.predict_next_days()
            df_next = df_next.rename(columns={'demand_predicted': 'Ontario Demand'})[['time','Ontario Demand']]
            df_next['month'] = pd.DatetimeIndex(df_next['time']).month
            monthly = pd.concat([monthly, df_next]).reset_index(drop=True)
            monthly['prob'] = self._get_percent_month(monthly)
            flag = list()
            top_demands = [quantile_80[monthly.month[0]-1]]
            monthly['percent'] = 0
            monthly['percent'] = (monthly['Ontario Demand'] / max_list[monthly.month[0]-1]) * 100
            for i in range(len(monthly)):
                if (monthly['Ontario Demand'][i] + 200 > sum(top_demands) / len(top_demands)) & (monthly.prob[i] == 85):
                    top_demands.append(monthly['Ontario Demand'][i])
                    if len(top_demands) > 5:
                        top_demands.sort(reverse=True)
                        top_demands = top_demands[:-1]
                    flag.append(True)
                else:
                    flag.append(False)
            monthly['flag'] = flag
            monthly.prob = np.where((monthly.flag == False) & (monthly.prob == 85), 80, monthly.prob)
            monthly.prob = np.where((monthly.flag == True) & (monthly.prob == 85), monthly.percent, monthly.prob)
            monthly.prob = np.where((monthly.prob < 80), monthly.percent - 10, monthly.prob)
            return monthly[monthly.time >= start].reset_index(drop=True)
#            return monthly.tail(7)
#             return monthly["percent"].iloc[-1]
        
        if data_x is not None:
            daily['Ontario Demand'] = np.where((daily.index==0), data_x, daily['Ontario Demand'])
            daily = self._prob_peak(daily)
            return daily["prob"].iloc[0]
        else:
            daily = self._prob_peak(daily)
            return daily["prob"].iloc[-1]
        
    def window(self, windows_number, predicted_demand = None):
        if predicted_demand is None:
            predicted_demand = self.daily_demand_prediction()
        sorted_demands = predicted_demand.sort_values('demand_predicted',ascending=False).head(windows_number).reset_index(drop=True).sort_values('time',ascending=True).reset_index(drop=True)
        sorted_demands['time_2'] = sorted_demands.time + timedelta(hours=-1)
        exact_peak = []
        inputs = []
        for i in range(windows_number):
            inputs.append([sorted_demands.time_2[i].hour,sorted_demands.time[i].hour])
            exact_peak.append(sorted_demands.time[i])
    
        intervals = [P.closed(a, b) for a, b in inputs]
    
        merge = P.Interval(*intervals)
        return merge
            
    
    



class Ontario_pipeline:
    
    def __init__(self):
        self.data = None
        self.et_model = None
        self.et_model_daily = None
        self.model_loaded = 0
        self.dt = Data()
        
    def _preprocess_x(self,daily=False):
        feature_engineering = FeatureEngineering()
        self.data = feature_engineering.ontario(self.data,daily)
        
    def _get_data(self,daily=False):
        self.data = pd.read_csv('raw_final_data.csv')
        
        if daily is not False:
            self.data['time'] = pd.to_datetime(self.data['time'])
            self.data = self.data.drop(['wind_speed'],axis=1)
            df = self.data[['time','Ontario Demand']]
            df.set_index('time',inplace=True)
            df = df.loc[df.groupby(pd.Grouper(freq='D')).idxmax().iloc[:, 0]].reset_index()[['time']]
            self.data = pd.merge(df, self.data, on='time', how='inner')
            self.data.drop('hour',axis=1,inplace=True)
            self.data = self.data.resample('24H', on = 'time').max()
            self.data.drop('time',axis=1,inplace=True)
            self.data.reset_index(inplace=True)

#         self.data = self.data[26305:]
        
#         self.data = self.data[26305:-8760]

        
    def train(self, n_estimators_n=5000, n_jobs_n=-1, learning_rate=0.03,
              boosting_type='dart',daily=False):
        

        self._get_data(daily)
        self._preprocess_x(daily)
        
        self.data.to_csv('test.csv')
        self._create_model(n_estimators_n, n_jobs_n, learning_rate, boosting_type,daily)
        return self.et_model
    
    def _create_model(self,n_estimators_n, n_jobs_n, learning_rate, boosting_type,daily):
        columns_name = self.data.columns
        filtered_columns = []
        for i in columns_name:
            if i != 'time' and i!= 'Ontario Demand':
                filtered_columns.append(i)
        X = self.data.drop(columns=['Ontario Demand','time'])
        y = self.data['Ontario Demand']
        transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='mean')),
                   ('scaler', StandardScaler())])
        preprocessor = ColumnTransformer(
            transformers=[('transform', transformer, filtered_columns)])
        pipeline = Pipeline(
            steps=[('preprocessor', preprocessor), 
                   ('regressor', lgb.LGBMRegressor(n_estimators=n_estimators_n,
                                                                n_jobs=n_jobs_n,
                                                                learning_rate=learning_rate,
                                                                boosting_type=boosting_type))])
        self.et_model = pipeline.fit(X, y)
        
        if daily is True:
            model_name = 'pipeline_model_ontario_daily.sav'
        else:
            model_name = 'pipeline_model_ontario.sav'
            
        joblib.dump(self.et_model, model_name, compress=3)
        
    def predict(self, dataframe:pd.DataFrame):
        if self.model_loaded == 0:
            self.et_model = joblib.load('pipeline_model_ontario.sav')
            self.model_loaded == 1
        self.data = dataframe
        self._preprocess_x()
        self.data['demand_predicted'] = self.et_model.predict(self.data)
        return self.data
    
    def predict_daily(self, dataframe:pd.DataFrame):
        if self.model_loaded == 0:
            self.et_model_daily = joblib.load('pipeline_model_ontario_daily.sav')
            self.model_loaded == 1
        self.data = dataframe
        self._preprocess_x(daily=True)
        self.data['demand_predicted'] = self.et_model_daily.predict(self.data)
        return self.data
    
    def predict_next_days(self,daily=True):
        daily_data = self.dt.daily_next_days()
        daily_predicted = self.predict_daily(daily_data)
        start_time = self.dt.toronto_time()
        start = datetime(start_time.year, start_time.month, start_time.day)
        end = start + timedelta(days=7)
        daily_predicted = daily_predicted[(daily_predicted.time > 
                         start)&(daily_predicted.time < end)].reset_index(drop=True)[['time','demand_predicted','temperature','ieso_demands']]
        return daily_predicted
