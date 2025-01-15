import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pvlib
from glob import glob
import os
import time

# Constants
LATITUDE = 28.009465
LONGITUDE = 72.980845
ALTITUDE = 217

class TimeSeriesLagModel:
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 6,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'device': 'cpu',
        }
        self.models = {}
        self.predictions = {}

    def prepare_data_train_range(self, X_train_df, y_train_df, new_stand_test, lag_features_dict, end_date=None, x_days=30):
        self.datasets = {}
        if end_date is None:
            end_date = X_train_df.index.max().strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end_date)
        start_date = end_date - pd.Timedelta(days=x_days)

        for lag_hours, features in lag_features_dict.items():
            mask_train = (X_train_df.index >= start_date)
            X_train_period = X_train_df[features][mask_train]
            y_train_period = y_train_df[mask_train]

            self.datasets[lag_hours] = {
                'dtrain': xgb.DMatrix(X_train_period.values, label=y_train_period.values),
                'dtest': xgb.DMatrix(new_stand_test[features].values)
            }

            self.date_ranges = {
                'start_date': start_date,
                'end_date': end_date
            }

    def train_models_no_eval(self, num_boost_round=1000, early_stopping_rounds=100, verbose_eval=100):
        for lag_hours, data in self.datasets.items():
            self.models[lag_hours] = xgb.train(
                self.params,
                data['dtrain'],
                num_boost_round=num_boost_round,
                evals=[(data['dtrain'], 'train')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval
            )

    def predict(self, test_index, night_hours_filter=True):
        self.predictions = {}
        for lag_hours, model in self.models.items():
            preds = pd.DataFrame(
                model.predict(self.datasets[lag_hours]['dtest']),
                index=pd.to_datetime(test_index)
            )
            if night_hours_filter:
                preds.loc[(preds.index.hour >= 19) | (preds.index.hour <= 6), 0] = 0
            self.predictions[lag_hours] = preds

class RollingPredictionSystem:
    def __init__(self, base_date, initial_data, lag_features_dict, start_hour=12, end_hour=23, x_days=30):
        self.base_date = pd.to_datetime(base_date)
        self.initial_data = initial_data
        self.lag_features = lag_features_dict
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.current_hour = start_hour
        self.predictions_history = {}
        self.rmse_history = {}
        self.prediction_plots = {}
        self.x_days = x_days

    def get_training_window(self, current_hour):
        cutoff_time = pd.Timestamp(f"{self.base_date.date()} {current_hour:02d}:00:00")
        return self.initial_data[self.initial_data.index <= cutoff_time]

    def get_prediction_window(self, current_hour):
        pred_start = current_hour + 2
        pred_start_time = pd.Timestamp(f"{self.base_date.date()} {pred_start:02d}:00:00")
        pred_end_time = pd.Timestamp(f"{self.base_date.date()} {self.end_hour:02d}:00:00")
        test_mask = (self.initial_data.index >= pred_start_time) & (self.initial_data.index <= pred_end_time)
        test_df = self.initial_data[test_mask]
        return test_df

    def plot_hour_predictions(self, hour, predictions_dict, actual_values):
        plt.figure(figsize=(12, 6))
        plt.plot(actual_values.index.hour, actual_values,
                label='Actual', color='black', linewidth=2)
        
        colors = {'2h': 'blue', '3h': 'green', '4h': 'red', '5h': 'purple'}
        
        for lag, preds in predictions_dict.items():
            if isinstance(preds, pd.DataFrame):
                preds = preds[0]
            
            lag_hours = int(lag[0])
            pred_hour = hour + lag_hours
            
            hour_pred = preds[preds.index.hour == pred_hour]
            if not hour_pred.empty:
                plt.plot(hour_pred.index.hour, hour_pred.values,
                        'o--', label=f'{lag} Lag', color=colors[lag],
                        markersize=8)

        plt.title(f'Predictions vs Actual at {hour}:00', fontsize=14)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Active Power', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(range(0, 24))
        plt.axvline(x=hour, color='gray', linestyle=':', label='Current Hour')
        plt.ylim(0, max(actual_values) * 1.1)
        plt.xlim(max(0, hour - 1), min(24, hour + 6))
        plt.tight_layout()
        return plt.gcf()

    def calculate_hour_rmse(self, hour, predictions_dict, actual_values):
        hour_rmse = {}
        for lag, preds in predictions_dict.items():
            if isinstance(preds, pd.DataFrame):
                preds = preds[0]
            
            lag_hours = int(lag[0])
            pred_hour = hour + lag_hours
            
            hour_pred = preds[preds.index.hour == pred_hour]
            hour_actual = actual_values[actual_values.index.hour == pred_hour]
            
            if not hour_pred.empty and not hour_actual.empty:
                hour_rmse[lag] = np.sqrt(mean_squared_error(
                    hour_actual,
                    hour_pred
                ))
        return hour_rmse

    def run_iteration(self):
        if (self.current_hour >= self.end_hour - 2) or (self.current_hour >= 16):
            return None, None, None, None

        training_data = self.get_training_window(self.current_hour)
        prediction_window = self.get_prediction_window(self.current_hour)

        X_train = training_data.drop(columns=['Active_Power'])
        y_train = training_data['Active_Power']

        ts_model = TimeSeriesLagModel()
        ts_model.prepare_data_train_range(
            X_train_df=X_train,
            y_train_df=y_train,
            new_stand_test=prediction_window,
            lag_features_dict=self.lag_features,
            x_days=self.x_days
        )
        ts_model.train_models_no_eval()
        ts_model.predict(test_index=prediction_window.index)

        hour_rmse = self.calculate_hour_rmse(
            self.current_hour,
            ts_model.predictions,
            prediction_window['Active_Power']
        )
        self.rmse_history[self.current_hour] = hour_rmse

        hour_plot = self.plot_hour_predictions(
            self.current_hour,
            ts_model.predictions,
            prediction_window['Active_Power']
        )
        self.prediction_plots[self.current_hour] = hour_plot

        self.predictions_history[self.current_hour] = ts_model.predictions
        self.current_hour += 1

        return self.current_hour - 1, ts_model.predictions, hour_rmse, hour_plot

    def run_all_iterations(self):
        results = {}
        while True:
            hour, predictions, rmse, plot = self.run_iteration()
            if hour is None:
                break
            results[hour] = {
                'predictions': predictions,
                'rmse': rmse,
                'plot': plot
            }
            print(f"\nCompleted predictions up to {hour}:00")
            print(f"RMSE values for hour {hour}:")
            for lag, value in rmse.items():
                print(f"  {lag} lag: {value:.4f}")

            plot.savefig(f'plots/predictions_hour_{hour}.png')
            plt.close()

        return results

def preprocess_data(df, latitude, longitude, altitude):

    df.rename(columns={'timestamp':'date'},inplace=True)
    df['date'] = pd.to_datetime(df['date']) 

    target_variable = 'Active_Power'

    standardize_predictor_list = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
      'apparent_temperature', 'rain', 'weather_code','pressure_msl', 'surface_pressure',
       'wind_gusts_10m', 'sunshine_duration','wet_bulb_temperature_2m',
       'terrestrial_radiation','surface_pressure', 'cloud_cover', 'wind_speed_10m',
       'wind_direction_10m', 'shortwave_radiation', 'direct_radiation',
       'diffuse_radiation', 'season', 'temperature_f', 'THI',
       'wind_speed_mph', 'wind_chill', 'heat_index', 'solar_zenith_angle',
       'air_mass', 'day_of_year', 'day_of_year_sin', 'day_of_year_cos',
       'hour_of_day', 'hour_of_day_sin', 'hour_of_day_cos',
       'Active_Power_lag_1h','Active_Power_lag_5h','Active_Power_lag_2h','Active_Power_lag_3h','Active_Power_lag_4h',
       'Active_Power_lag_21h','Active_Power_lag_22h','Active_Power_lag_23h','Active_Power_lag_24h',
       'pressure_msl_lag_1h','pressure_msl_lag_2h','pressure_msl_lag_3h','pressure_msl_lag_4h',
       'pressure_msl_lag_21h','pressure_msl_lag_22h','pressure_msl_lag_23h','pressure_msl_lag_24h',
       'direct_radiation_lag_1h','direct_radiation_lag_2h','direct_radiation_lag_3h','direct_radiation_lag_4h',
       'direct_radiation_lag_21h','direct_radiation_lag_22h','direct_radiation_lag_23h','direct_radiation_lag_24h',
       'cloud_cover_lag_1h','cloud_cover_lag_2h','cloud_cover_lag_3h','cloud_cover_lag_4h',
       'cloud_cover_lag_21h','cloud_cover_lag_22h','cloud_cover_lag_23h','cloud_cover_lag_24h',
       'temperature_2m_lag_1h','temperature_2m_lag_2h','temperature_2m_lag_3h','temperature_2m_lag_4h',
       'temperature_2m_lag_21h','temperature_2m_lag_22h','temperature_2m_lag_23h','temperature_2m_lag_24h',
      'wind_speed_10m_lag_1h','wind_speed_10m_lag_2h','wind_speed_10m_lag_3h','wind_speed_10m_lag_4h',
       'wind_speed_10m_lag_21h','wind_speed_10m_lag_22h','wind_speed_10m_lag_23h','wind_speed_10m_lag_24h',
       'temperature_2m_lag_1h','temperature_2m_lag_24h', 'cloud_cover_lag_1h', 'cloud_cover_lag_24h',
       'wind_speed_10m_lag_1h', 'wind_speed_10m_lag_24h',
       'temperature_2m_rolling_mean_24h', 'temperature_2m_rolling_std_24h',
       'cloud_cover_rolling_mean_24h', 'cloud_cover_rolling_std_24h',
       'wind_speed_10m_rolling_mean_24h', 'wind_speed_10m_rolling_std_24h',
       'temperature_2m_rolling_mean_2h', 'temperature_2m_rolling_std_2h',
       'cloud_cover_rolling_mean_2h', 'cloud_cover_rolling_std_2h',
       'wind_speed_10m_rolling_mean_2h', 'wind_speed_10m_rolling_std_2h',
       'temperature_2m_rolling_mean_3h', 'temperature_2m_rolling_std_3h',
       'cloud_cover_rolling_mean_3h', 'cloud_cover_rolling_std_3h',
       'wind_speed_10m_rolling_mean_3h', 'wind_speed_10m_rolling_std_3h',
        'temperature_2m_rolling_mean_4h', 'temperature_2m_rolling_std_4h',
       'cloud_cover_rolling_mean_4h', 'cloud_cover_rolling_std_4h',
       'wind_speed_10m_rolling_mean_4h', 'wind_speed_10m_rolling_std_4h',
       'temp_wind_interaction', 'cloud_radiation_interaction',
       'temperature_2m_change', 'cloud_cover_change', 'wind_speed_10m_change',
       'weather_stability_index','pressure_msl_rolling_mean_24h', 'pressure_msl_rolling_std_24h',
       'pressure_msl_rolling_mean_2h', 'pressure_msl_rolling_std_2h',
       'pressure_msl_rolling_mean_3h', 'pressure_msl_rolling_std_3h',
       'pressure_msl_rolling_mean_4h', 'pressure_msl_rolling_std_4h',
        'direct_radiation_rolling_mean_24h', 'direct_radiation_rolling_std_24h',
       'direct_radiation_rolling_mean_2h', 'direct_radiation_rolling_std_2h',
       'direct_radiation_rolling_mean_3h', 'direct_radiation_rolling_std_3h',
       'direct_radiation_rolling_mean_4h', 'direct_radiation_rolling_std_4h',
       ]

    # Add season
    def add_season(df):
        def season(month):
            if month in [12,1,2,3]:
                return 'Winter'
            elif month in [10,11]:
                return 'Post-Monsoon'
            elif month in [6,7,8,9]:
                return 'Monsoon'
            else:
                return 'Summer'
        df['season'] = df['date'].dt.month.apply(season)
        return df

    df = add_season(df)

    df = df.set_index('date')
    df = df.sort_values('date')

    ord_enc = OrdinalEncoder()
    season = ord_enc.fit(np.array(df['season']).reshape(-1,1))
    season_train = ord_enc.transform(np.array(df['season']).reshape(-1,1))
    df['season'] = season_train

    # Set Active Power to 0 during night hours
    df.loc[((df.index.hour >= 19) | (df.index.hour <= 6)) | (df['Active_Power'] < 0), 'Active_Power'] = 0

    test_date = '2024-12-04'
    test = df.loc[test_date]
    train = df

    def standardize_data(new_train, new_test):
        X_new_train = new_train[standardize_predictor_list]
        X_new_test = new_test[standardize_predictor_list]
        predictor_scaler = StandardScaler()
        predictor_scaler_fit = predictor_scaler.fit(X_new_train)

        X_new_train= predictor_scaler_fit.transform(X_new_train)
        X_new_test = predictor_scaler_fit.transform(X_new_test)

        joblib.dump(predictor_scaler_fit, f'predictor_scaler_fit.pkl')

        new_stand_train = pd.DataFrame(X_new_train, index=new_train[standardize_predictor_list].index, columns=new_train[standardize_predictor_list].columns)
        new_stand_test = pd.DataFrame(X_new_test, index=new_test[standardize_predictor_list].index, columns=new_test[standardize_predictor_list].columns)

        categorical_columns = ['time_interval']
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoded_features = encoder.fit(new_train[categorical_columns])
        encoded_features_train = encoder.transform(new_train[categorical_columns])
        encoded_features_test = encoder.transform(new_test[categorical_columns])
        encoded_train = pd.DataFrame(encoded_features_train, columns=categorical_columns, index=new_train.index)
        encoded_test = pd.DataFrame(encoded_features_test, columns=categorical_columns, index=new_test.index)

        joblib.dump(encoded_features, f'encoded_features.pkl')

        new_stand_train = pd.concat([new_stand_train, encoded_train , new_train['Active_Power']], axis = 1)
        new_stand_test = pd.concat([new_stand_test, encoded_test], axis = 1)

        return new_stand_train, new_stand_test

    # Detect time interval
    # Detect time interval
    def detect_time_interval(df):
        df_time_detect = df.copy()
        intervals = {'zeroth_interval': (0,6) , 'first_interval': (7, 9), 'second_interval': (9, 11), 'third_interval': (11, 13),
                    'fourth_interval': (13, 15), 'fifth_interval': (15, 17), 'sixth_interval': (17, 18) , 'seventh_interval': (19, 23) }
        df_time_detect['time_interval'] = pd.cut(df_time_detect.index.hour, bins=[interval[0] for interval in intervals.values()] + [24],
                                    labels=[interval_name for interval_name in intervals.keys()],
                                    include_lowest=True, right=False)
        return df_time_detect
    
    def nonlinear_features(df,latitude,longitude,altitude):
        # Ensure the dataframe has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            df.set_index('datetime', inplace=True)

        # 2. Temperature-Humidity Index (THI)
        df['temperature_f'] = df['temperature_2m'] * 9/5 + 32  # Convert to Fahrenheit
        df['THI'] = df['temperature_f'] - (0.55 - 0.0055 * df['relative_humidity_2m']) * (df['temperature_f'] - 58)

        # 3. Wind Chill Factor
        df['wind_speed_mph'] = df['wind_speed_10m'] * 2.237  # Convert to mph
        df['wind_chill'] = 35.74 + 0.6215*df['temperature_f'] - 35.75*(df['wind_speed_mph']**0.16) + 0.4275*df['temperature_f']*(df['wind_speed_mph']**0.16)

        # 4. Heat Index
        df['heat_index'] = -42.379 + 2.04901523*df['temperature_f'] + 10.14333127*df['relative_humidity_2m'] - 0.22475541*df['temperature_f']*df['relative_humidity_2m'] - 6.83783e-3*df['temperature_f']**2 - 5.481717e-2*df['relative_humidity_2m']**2 + 1.22874e-3*df['temperature_f']**2*df['relative_humidity_2m'] + 8.5282e-4*df['temperature_f']*df['relative_humidity_2m']**2 - 1.99e-6*df['temperature_f']**2*df['relative_humidity_2m']**2

        # 5. Solar Zenith Angle
        df['solar_zenith_angle'] = calculate_solar_zenith_angle(df, latitude, longitude, altitude)

        # 6. Air Mass
        df['air_mass'] = 1 / np.cos(np.radians(df['solar_zenith_angle']))

        # 9. Day of Year Sine and Cosine
        df['day_of_year'] = df.index.dayofyear
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

        # 10. Hour of Day Sine and Cosine
        df['hour_of_day'] = df.index.hour + df.index.minute / 60
        df['hour_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

        # 11. Lagged Variables
        for col in ['Active_Power','pressure_msl','direct_radiation','temperature_2m', 'cloud_cover', 'wind_speed_10m']:
            df[f'{col}_lag_1h'] = df[col].shift(1)
            df[f'{col}_lag_2h'] = df[col].shift(2)
            df[f'{col}_lag_3h'] = df[col].shift(3)
            df[f'{col}_lag_4h'] = df[col].shift(4)
            df[f'{col}_lag_21h'] = df[col].shift(21)
            df[f'{col}_lag_22h'] = df[col].shift(22)
            df[f'{col}_lag_23h'] = df[col].shift(23)
            df[f'{col}_lag_24h'] = df[col].shift(24)

        df[f'Active_Power_lag_5h'] = df['Active_Power'].shift(5)

        # 11. Lagged Variables
        for col in ['temperature_2m', 'cloud_cover', 'wind_speed_10m']:
            df[f'{col}_lag_1h'] = df[col].shift(1)
            df[f'{col}_lag_24h'] = df[col].shift(24)

        # 12. Rolling Statistics
        for col in ['pressure_msl','direct_radiation','temperature_2m', 'cloud_cover', 'wind_speed_10m']:
            df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24).mean()
            df[f'{col}_rolling_std_24h'] = df[col].rolling(window=24).std()
            df[f'{col}_rolling_mean_2h'] = df[col].rolling(window=2).mean()
            df[f'{col}_rolling_std_2h'] = df[col].rolling(window=2).std()
            df[f'{col}_rolling_mean_3h'] = df[col].rolling(window=3).mean()
            df[f'{col}_rolling_std_3h'] = df[col].rolling(window=3).std()
            df[f'{col}_rolling_mean_4h'] = df[col].rolling(window=4).mean()
            df[f'{col}_rolling_std_4h'] = df[col].rolling(window=4).std()

        # 13. Interaction Terms
        df['temp_wind_interaction'] = df['temperature_2m'] * df['wind_speed_10m']
        df['cloud_radiation_interaction'] = df['cloud_cover'] * df['direct_radiation']

        # 15. Weather Stability Index
        # Calculate the change in key weather variables
        for col in ['temperature_2m', 'cloud_cover', 'wind_speed_10m']:
            df[f'{col}_change'] = df[col].diff()

        # Create a composite weather stability index
        df['weather_stability_index'] = (df['temperature_2m_change'].abs() +
                                        df['cloud_cover_change'].abs() +
                                        df['wind_speed_10m_change'].abs())

        return df

    def calculate_solar_zenith_angle(df, latitude, longitude, altitude):
        site = pvlib.location.Location(latitude, longitude, altitude=altitude)

        solar_position = site.get_solarposition(df.index)
        solar_zenith_angle = solar_position['zenith']

        return solar_zenith_angle
    
    train = detect_time_interval(train)
    test = detect_time_interval(test)

    non_linear_train = train.copy()
    non_linear_test = test.copy()

    non_linear_train = nonlinear_features(non_linear_train,latitude,longitude,altitude) 
    non_linear_test = nonlinear_features(non_linear_test,latitude,longitude,altitude)

    non_linear_train.bfill(inplace=True)
    non_linear_test.bfill(inplace=True)

    # Calculate solar position
    def calculate_solar_zenith_angle(df, latitude, longitude, altitude):
        site = pvlib.location.Location(latitude, longitude, altitude=altitude)
        solar_position = site.get_solarposition(df.index)
        return solar_position['zenith']

    new_stand_train, new_stand_test = standardize_data(non_linear_train, non_linear_test)

    return new_stand_train, new_stand_test

def create_dashboard():
    # st.title("Solar Power Prediction Dashboard")
    st.subheader("Predictions & RMSE Analysis for Dec 4, 2024: Impact of Time Horizons & Data Durations")
    
    # Load and preprocess data
    df = pd.read_csv("seci2024_mar_dec_weather.csv")
    processed_df, processed_df_test = preprocess_data(df, LATITUDE, LONGITUDE, ALTITUDE)
    processed_df.fillna(method='bfill', inplace=True)

    # Create placeholders for real-time updates
    button_placeholder = st.empty()
    status_placeholder = st.empty()
    plot_placeholder = st.empty()
    explanation_placeholder = st.empty()
    
    # Initialize session state if not exists
    if 'current_hour' not in st.session_state:
        st.session_state.current_hour = 6
    if 'predictions_complete' not in st.session_state:
        st.session_state.predictions_complete = False
    if 'is_predicting' not in st.session_state:
        st.session_state.is_predicting = False

    # Show appropriate button based on state
    if not st.session_state.is_predicting and not st.session_state.predictions_complete:
        if button_placeholder.button("Generate New Forecast", type="primary"):
            st.session_state.is_predicting = True
            st.rerun()

    elif st.session_state.predictions_complete:
        if button_placeholder.button("Show All Hours Predictions", type="primary"):
            st.session_state.show_all = True
            st.rerun()

    feat_imp_lag_2h = ['hour_of_day_cos',
    'Active_Power_lag_2h',
    'direct_radiation_lag_4h',
    'Active_Power_lag_24h',
    'day_of_year_cos',
    'Active_Power_lag_3h',
    'direct_radiation_lag_3h',
    'Active_Power_lag_22h',
    'Active_Power_lag_21h',
    'day_of_year_sin',
    'direct_radiation_rolling_std_4h',
    'Active_Power_lag_23h',
    'temperature_2m_change',
    'Active_Power_lag_5h',
    'pressure_msl_rolling_mean_24h',
    'direct_radiation_lag_2h',
    'cloud_cover_rolling_std_4h',
    'hour_of_day',
    'air_mass',
    'pressure_msl_rolling_std_4h',
    'wind_speed_10m_lag_3h',
    'Active_Power_lag_4h',
    'temperature_2m_rolling_std_24h',
    'pressure_msl',
    'wind_speed_10m_lag_22h']

    feat_imp_lag_3h = ['hour_of_day_cos',
    'Active_Power_lag_3h',
    'direct_radiation_lag_4h',
    'Active_Power_lag_24h',
    'day_of_year_cos',
    'Active_Power_lag_22h',
    'direct_radiation_rolling_std_4h',
    'Active_Power_lag_21h',
    'day_of_year_sin',
    'Active_Power_lag_4h',
    'direct_radiation_lag_3h',
    'day_of_year',
    'Active_Power_lag_23h',
    'pressure_msl',
    'direct_radiation_rolling_std_24h',
    'pressure_msl_rolling_mean_24h',
    'THI',
    'temperature_2m_lag_3h',
    'time_interval',
    'pressure_msl_lag_3h',
    'Active_Power_lag_5h',
    'solar_zenith_angle',
    'temperature_2m',
    'pressure_msl_rolling_std_4h',
    'direct_radiation_rolling_std_3h']

    feat_imp_lag_4h = ['hour_of_day_cos',
    'direct_radiation_lag_4h',
    'Active_Power_lag_24h',
    'Active_Power_lag_4h',
    'day_of_year_cos',
    'direct_radiation_lag_3h',
    'Active_Power_lag_21h',
    'day_of_year_sin',
    'day_of_year',
    'Active_Power_lag_5h',
    'time_interval',
    'Active_Power_lag_22h',
    'direct_radiation_rolling_std_4h',
    'direct_radiation_rolling_std_24h',
    'temperature_2m_lag_4h',
    'solar_zenith_angle',
    'Active_Power_lag_23h',
    'temperature_2m_rolling_mean_24h',
    'direct_radiation_rolling_mean_24h',
    'relative_humidity_2m',
    'pressure_msl_rolling_mean_24h',
    'THI',
    'air_mass',
    'wind_direction_10m',
    'wind_speed_10m_lag_23h']

    feat_imp_lag_5h = ['hour_of_day_cos',
    'direct_radiation_lag_4h',
    'Active_Power_lag_24h',
    'day_of_year_cos',
    'Active_Power_lag_5h',
    'day_of_year',
    'day_of_year_sin',
    'direct_radiation_rolling_std_4h',
    'Active_Power_lag_22h',
    'Active_Power_lag_21h',
    'time_interval',
    'air_mass',
    'solar_zenith_angle',
    'direct_radiation_rolling_std_24h',
    'temperature_2m_lag_3h',
    'temperature_2m_lag_4h',
    'direct_radiation_lag_3h',
    'Active_Power_lag_23h',
    'direct_radiation_rolling_mean_24h',
    'wind_direction_10m',
    'temperature_2m',
    'THI',
    'direct_radiation_lag_22h',
    'pressure_msl_lag_3h',
    'wind_speed_10m_lag_23h']

    # Run predictions if in predicting state
    if st.session_state.is_predicting and not st.session_state.predictions_complete:
        lag_features = {
            '2h': feat_imp_lag_2h,
            '3h': feat_imp_lag_3h,
            '4h': feat_imp_lag_4h,
            '5h': feat_imp_lag_5h
        }

        # Initialize prediction system
        rps = RollingPredictionSystem(
            base_date='2024-12-04',
            initial_data=processed_df,
            lag_features_dict=lag_features,
            start_hour=6,
            end_hour=23,
            x_days=15
        )

        # Run predictions hour by hour
        while True:
            status_placeholder.text(f"Generating predictions for {st.session_state.current_hour}:00...")
            
            hour, predictions, rmse, plot = rps.run_iteration()
            
            if hour is None:
                st.session_state.predictions_complete = True
                st.session_state.is_predicting = False
                status_placeholder.text("All predictions complete!")
                st.rerun()
                break
            
            # Save the plot
            plot_path = f'plots/predictions_hour_{hour}.png'
            plot.savefig(plot_path)
            plt.close()

            # Update display
            plot_placeholder.image(plot_path, caption=f"Predictions vs Actual at {hour:02d}:00")
            
            # Show explanation
            explanation_placeholder.markdown("""
            ### Plot Explanation:
            - **Black line**: Actual power generation values
            - **Blue markers**: 2-hour ahead predictions
            - **Green markers**: 3-hour ahead predictions
            - **Red markers**: 4-hour ahead predictions
            - **Purple markers**: 5-hour ahead predictions
            - **Vertical dotted line**: Current prediction hour
            """)

            # # Print RMSE values
            # st.write(f"RMSE values for hour {hour}:")
            # for lag, value in rmse.items():
            #     st.write(f"  {lag} lag: {value:.4f}")
            
            st.session_state.current_hour = hour + 1
            time.sleep(0.5)  # Add small delay for visualization
    
    # Show all predictions view
    if st.session_state.predictions_complete and getattr(st.session_state, 'show_all', False):
        plot_files = glob('plots/predictions_hour_*.png')
        available_hours = sorted([int(f.split('_')[-1].split('.')[0]) for f in plot_files])
        
        selected_hour = st.selectbox(
            "Select Prediction Hour",
            available_hours,
            format_func=lambda x: f"{x:02d}:00"
        )

        if selected_hour:
            plot_path = f'plots/predictions_hour_{selected_hour}.png'
            if os.path.exists(plot_path):
                plot_placeholder.image(plot_path, caption=f"Predictions vs Actual at {selected_hour:02d}:00")
                
                # Add download button
                with open(plot_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Plot",
                        data=file,
                        file_name=f"solar_prediction_{selected_hour}h.png",
                        mime="image/png"
                    )

                # Show additional plots in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.image('all_hr_curves.png', caption='Comparison of predictions made at different hour', use_container_width=True)
                    st.image('lag_rmse.png', caption='RMSE comparison by Lag Time', use_container_width=True)
                with col2:
                    st.image('final_predictions.png', caption='Final Predictions', use_container_width=True)
                    st.image('time_rmse.png', caption='RMSE comparison across Training Periods', use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Solar Power Prediction Dashboard",
        page_icon="☀️",
        layout="wide"
    )

    # Add the image at the top-left
    col1, col2 = st.columns([1,8])  # Adjust column widths

    # Replace 'your_image_path.png' with the path to your image
    with col1:
        st.image("teqo-logo-1.png", use_container_width=True)  # Display the image
    with col2:
        st.title("Solar Power Prediction Dashboard") 
    
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
        }
        .reportview-container {
            margin-top: -2em;
        }
        .stSelectbox {
            margin-bottom: 2em;
        }
        </style>
        """, unsafe_allow_html=True)
    
    create_dashboard()

