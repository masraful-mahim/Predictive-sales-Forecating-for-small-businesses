# Import libraries
import pandas as pd
import numpy as np
import holidays
from datetime import timedelta
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess data
def load_and_preprocess_data():
    try:
        # Load dataset
        df = pd.read_csv(r"D:\Projects\Predictive sales Forecating for small businesses\Sales Dataset.csv")
        
        # Convert Order Date to datetime
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        
        # Create time-based features
        df['DayOfWeek'] = df['Order Date'].dt.dayofweek
        df['Month'] = df['Order Date'].dt.month
        df['Year'] = df['Order Date'].dt.year
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Add US holidays
        us_holidays = holidays.US(years=range(2020, 2026))
        df['IsHoliday'] = df['Order Date'].apply(lambda x: 1 if x in us_holidays else 0)
        
        # One-hot encode categorical variables
        categorical_cols = ['Category', 'PaymentMode', 'State']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
        
        # Aggregate by date
        agg_dict = {
            'Amount': 'sum',
            'Quantity': 'sum',
            'Profit': 'sum',
            'DayOfWeek': 'first',
            'Month': 'first',
            'Year': 'first',
            'IsWeekend': 'first',
            'IsHoliday': 'first'
        }
        # Add one-hot encoded columns to aggregation
        for col in df_encoded.columns:
            if any(prefix in col for prefix in categorical_cols):
                agg_dict[col] = 'sum'  # Sum the one-hot encoded columns
        
        agg_df = df_encoded.groupby('Order Date').agg(agg_dict).reset_index()
        
        # Fill NaN values in one-hot encoded columns with 0
        for col in agg_df.columns:
            if any(prefix in col for prefix in categorical_cols):
                agg_df[col] = agg_df[col].fillna(0)
        
        # Fill NaN values in other columns (e.g., Profit, Quantity) with forward/backward fill
        agg_df = agg_df.fillna(method='ffill').fillna(method='bfill')
        
        # Verify no NaNs remain
        if agg_df.isna().any().any():
            print("NaN values found after preprocessing:")
            print(agg_df.isna().sum())
            raise ValueError("NaN values found after preprocessing")
        
        return df, agg_df
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return None, None

# Anomaly detection using IQR
def detect_anomalies(df, column='Amount'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['IsAnomaly'] = df[column].apply(lambda x: 1 if x < lower_bound or x > upper_bound else 0)
    return df

# Train models and generate forecasts
def train_models_and_forecast(agg_df):
    try:
        # Split data
        train_size = int(len(agg_df) * 0.8)
        train_df = agg_df.iloc[:train_size]
        test_df = agg_df.iloc[train_size:]
        
        # Prophet model
        prophet_df = train_df.rename(columns={'Order Date': 'ds', 'Amount': 'y'})
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        prophet_model.add_country_holidays(country_name='US')
        regressors = ['Quantity', 'Profit', 'DayOfWeek', 'Month', 'Year', 'IsWeekend', 'IsHoliday'] + \
                     [col for col in agg_df.columns if 'Category_' in col or 'PaymentMode_' in col or 'State_' in col]
        for regressor in regressors:
            prophet_model.add_regressor(regressor)
        
        # Check for NaN values in prophet_df
        if prophet_df[regressors].isna().any().any():
            raise ValueError(f"NaN values found in regressors: {prophet_df[regressors].isna().sum()}")
        
        prophet_model.fit(prophet_df)
        
        # Prophet forecast
        future = prophet_model.make_future_dataframe(periods=len(test_df), freq='D')
        future = future.merge(agg_df[['Order Date'] + regressors], left_on='ds', right_on='Order Date', how='left')
        future = future.drop(columns=['Order Date']).fillna(method='ffill').fillna(method='bfill')
        
        # Ensure no NaN values in future dataframe
        if future[regressors].isna().any().any():
            raise ValueError(f"NaN values found in future regressors: {future[regressors].isna().sum()}")
        
        prophet_forecast = prophet_model.predict(future)
        prophet_rmse = np.sqrt(mean_squared_error(test_df['Amount'], prophet_forecast['yhat'].iloc[train_size:]))
        
        # ARIMA model
        arima_model = ARIMA(train_df['Amount'], order=(5, 1, 0))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=len(test_df))
        arima_rmse = np.sqrt(mean_squared_error(test_df['Amount'], arima_forecast))
        
        # Future 30-day forecast
        future_dates = pd.date_range(start=agg_df['Order Date'].max() + timedelta(days=1), periods=30, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        for col in regressors:
            future_df[col] = agg_df[col].iloc[-1]  # Use last known values
        future_df['DayOfWeek'] = future_df['ds'].dt.dayofweek
        future_df['Month'] = future_df['ds'].dt.month
        future_df['Year'] = future_df['ds'].dt.year
        future_df['IsWeekend'] = future_df['DayOfWeek'].isin([5, 6]).astype(int)
        future_df['IsHoliday'] = future_df['ds'].apply(lambda x: 1 if x in holidays.US(years=range(2020, 2026)) else 0)
        
        # Fill any remaining NaN values in future_df
        future_df = future_df.fillna(method='ffill').fillna(method='bfill')
        
        prophet_future = prophet_model.predict(future_df)
        arima_future = arima_fit.forecast(steps=30)
        
        return prophet_model, prophet_forecast, prophet_rmse, arima_fit, arima_forecast, arima_rmse, prophet_future, arima_future
    except Exception as e:
        print(f"Error in training models: {e}")
        return None, None, None, None, None, None, None, None

# Category-specific forecasts
def category_forecasts(df):
    try:
        category_models = {}
        category_forecasts = {}
        for category in df['Category'].unique():
            cat_df = df[df['Category'] == category].groupby('Order Date')['Amount'].sum().reset_index()
            prophet_df = cat_df.rename(columns={'Order Date': 'ds', 'Amount': 'y'})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=30, freq='D')
            forecast = model.predict(future)
            category_models[category] = model
            category_forecasts[category] = forecast
        return category_models, category_forecasts
    except Exception as e:
        print(f"Error in category forecasts: {e}")
        return None, None

# Initialize Dash app
app = Dash(__name__)
app.title = "Sales Forecasting Dashboard"

# Load data and train models
df, agg_df = load_and_preprocess_data()
if agg_df is not None:
    agg_df = detect_anomalies(agg_df)
    prophet_model, prophet_forecast, prophet_rmse, arima_fit, arima_forecast, arima_rmse, prophet_future, arima_future = train_models_and_forecast(agg_df)
    if prophet_model is None:
        print("Failed to train models. Exiting.")
        exit(1)
    category_models, category_forecasts = category_forecasts(df)
else:
    print("Failed to load and preprocess data. Exiting.")
    exit(1)

# Dashboard layout
app.layout = html.Div([
    # Theme toggle
    html.Div([
        dcc.Dropdown(
            id='theme-toggle',
            options=[
                {'label': 'Light', 'value': 'light'},
                {'label': 'Dark', 'value': 'dark'}
            ],
            value='light',
            clearable=False
        )
    ], style={'width': '10%', 'display': 'inline-block'}),
    
    # Summary statistics
    html.Div([
        html.H3("Summary Statistics"),
        html.P(id='total-sales'),
        html.P(id='prophet-rmse'),
        html.P(id='arima-rmse'),
        html.Button("Download Report", id='download-report-btn'),
        dcc.Download(id='download-report')
    ]),
    
    # Daily Sales Over Time Plot
    html.Div([
        html.H3("Daily Sales Over Time"),
        dcc.DatePickerRange(
            id='date-picker',
            min_date_allowed=agg_df['Order Date'].min(),
            max_date_allowed=agg_df['Order Date'].max(),
            initial_visible_month=agg_df['Order Date'].max(),
            start_date=agg_df['Order Date'].min(),
            end_date=agg_df['Order Date'].max()
        ),
        dcc.Graph(id='daily-sales-plot')
    ]),
    
    # Total Sales by Category Plot
    html.Div([
        html.H3("Total Sales by Category"),
        dcc.Dropdown(
            id='state-filter',
            options=[{'label': s, 'value': s} for s in df['State'].unique()],
            value=None,
            placeholder="Select a State"
        ),
        dcc.Graph(id='category-sales-plot')
    ]),
    
    # Forecast vs Actual Plot
    html.Div([
        html.H3("Forecast vs Actual"),
        dcc.Dropdown(
            id='model-selector',
            options=[
                {'label': 'Prophet', 'value': 'prophet'},
                {'label': 'ARIMA', 'value': 'arima'},
                {'label': 'Both', 'value': 'both'}
            ],
            value='both',
            clearable=False
        ),
        dcc.Graph(id='forecast-plot')
    ]),
    
    # Future Sales Forecast Plot
    html.Div([
        html.H3("Future Sales Forecast"),
        dcc.Slider(
            id='confidence-slider',
            min=0.5,
            max=0.99,
            step=0.01,
            value=0.95,
            marks={i: f'{i:.2f}' for i in np.arange(0.5, 1.0, 0.1)}
        ),
        dcc.Graph(id='future-forecast-plot'),
        html.Button("Download Forecast", id='download-forecast-btn'),
        dcc.Download(id='download-forecast')
    ]),
    
    # Category-Specific Forecasts Plot
    html.Div([
        html.H3("Category-Specific Forecasts"),
        dcc.Dropdown(
            id='category-selector',
            options=[{'label': c, 'value': c} for c in df['Category'].unique()],
            value=df['Category'].unique()[0],
            clearable=False
        ),
        dcc.Graph(id='category-forecast-plot'),
        html.Button("Download Category Forecast", id='download-category-btn'),
        dcc.Download(id='download-category')
    ])
], id='main-div')

# Callbacks
@app.callback(
    Output('main-div', 'style'),
    Input('theme-toggle', 'value')
)
def update_theme(theme):
    if theme == 'dark':
        return {'backgroundColor': '#1a1a1a', 'color': '#ffffff'}
    return {'backgroundColor': '#ffffff', 'color': '#000000'}

@app.callback(
    [Output('total-sales', 'children'),
     Output('prophet-rmse', 'children'),
     Output('arima-rmse', 'children')],
    Input('theme-toggle', 'value')  # Dummy input to trigger on load
)
def update_summary(_):
    total_sales = f"Total Sales: ${agg_df['Amount'].sum():,.2f}"
    prophet_rmse_text = f"Prophet RMSE: {prophet_rmse:,.2f}"
    arima_rmse_text = f"ARIMA RMSE: {arima_rmse:,.2f}"
    return total_sales, prophet_rmse_text, arima_rmse_text

@app.callback(
    Output('download-report', 'data'),
    Input('download-report-btn', 'n_clicks'),
    prevent_initial_call=True
)
def download_report(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    pdf_file = "sales_report.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"Total Sales: ${agg_df['Amount'].sum():,.2f}", styles['Heading1']),
        Paragraph(f"Prophet RMSE: {prophet_rmse:,.2f}", styles['Normal']),
        Paragraph(f"ARIMA RMSE: {arima_rmse:,.2f}", styles['Normal']),
        Spacer(1, 12)
    ]
    doc.build(story)
    return dcc.send_file(pdf_file)

@app.callback(
    Output('daily-sales-plot', 'figure'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('theme-toggle', 'value')]
)
def update_daily_sales(start_date, end_date, theme):
    filtered_df = agg_df[(agg_df['Order Date'] >= start_date) & (agg_df['Order Date'] <= end_date)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['Order Date'], y=filtered_df['Amount'], mode='lines', name='Sales'))
    anomalies = filtered_df[filtered_df['IsAnomaly'] == 1]
    fig.add_trace(go.Scatter(x=anomalies['Order Date'], y=anomalies['Amount'], mode='markers', name='Anomalies', marker=dict(color='red')))
    fig.update_layout(title="Daily Sales Over Time", template='plotly_dark' if theme == 'dark' else 'plotly')
    return fig

@app.callback(
    Output('category-sales-plot', 'figure'),
    [Input('state-filter', 'value'),
     Input('theme-toggle', 'value')]
)
def update_category_sales(state, theme):
    filtered_df = df if state is None else df[df['State'] == state]
    cat_sales = filtered_df.groupby('Category')['Amount'].sum().reset_index()
    fig = px.bar(cat_sales, x='Category', y='Amount', title="Total Sales by Category")
    fig.update_layout(template='plotly_dark' if theme == 'dark' else 'plotly')
    return fig

@app.callback(
    Output('forecast-plot', 'figure'),
    [Input('model-selector', 'value'),
     Input('theme-toggle', 'value')]
)
def update_forecast_plot(model, theme):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg_df['Order Date'], y=agg_df['Amount'], mode='lines', name='Actual'))
    if model in ['prophet', 'both']:
        fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], mode='lines', name='Prophet'))
        fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_upper'], mode='lines', name='Upper CI', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_lower'], mode='lines', name='Lower CI', line=dict(dash='dash')))
    if model in ['arima', 'both']:
        fig.add_trace(go.Scatter(x=agg_df['Order Date'].iloc[-len(arima_forecast):], y=arima_forecast, mode='lines', name='ARIMA'))
    fig.update_layout(title="Forecast vs Actual", template='plotly_dark' if theme == 'dark' else 'plotly')
    return fig

@app.callback(
    Output('future-forecast-plot', 'figure'),
    [Input('confidence-slider', 'value'),
     Input('theme-toggle', 'value')]
)
def update_future_forecast(confidence, theme):
    prophet_model.changepoint_prior_scale = 0.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prophet_future['ds'], y=prophet_future['yhat'], mode='lines', name='Prophet Forecast'))
    fig.add_trace(go.Scatter(x=prophet_future['ds'], y=prophet_future['yhat_upper'], mode='lines', name='Upper CI', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=prophet_future['ds'], y=prophet_future['yhat_lower'], mode='lines', name='Lower CI', line=dict(dash='dash')))
    fig.update_layout(title="Future Sales Forecast", template='plotly_dark' if theme == 'dark' else 'plotly')
    return fig

@app.callback(
    Output('download-forecast', 'data'),
    Input('download-forecast-btn', 'n_clicks'),
    prevent_initial_call=True
)
def download_forecast(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    forecast_df = pd.DataFrame({
        'Date': prophet_future['ds'],
        'Prophet_Forecast': prophet_future['yhat'],
        'Prophet_Upper': prophet_future['yhat_upper'],
        'Prophet_Lower': prophet_future['yhat_lower'],
        'ARIMA_Forecast': arima_future
    })
    return dcc.send_data_frame(forecast_df.to_csv, "future_forecast.csv")

@app.callback(
    Output('category-forecast-plot', 'figure'),
    [Input('category-selector', 'value'),
     Input('theme-toggle', 'value')]
)
def update_category_forecast(category, theme):
    forecast = category_forecasts[category]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name=f'{category} Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper CI', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower CI', line=dict(dash='dash')))
    fig.update_layout(title=f"Forecast for {category}", template='plotly_dark' if theme == 'dark' else 'plotly')
    return fig

@app.callback(
    Output('download-category', 'data'),
    [Input('download-category-btn', 'n_clicks'),
     Input('category-selector', 'value')],
    prevent_initial_call=True
)
def download_category_forecast(n_clicks, category):
    if n_clicks is None:
        raise PreventUpdate
    forecast = category_forecasts[category]
    forecast_df = pd.DataFrame({
        'Date': forecast['ds'],
        'Forecast': forecast['yhat'],
        'Upper': forecast['yhat_upper'],
        'Lower': forecast['yhat_lower']
    })
    return dcc.send_data_frame(forecast_df.to_csv, f"{category}_forecast.csv")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)  # Updated from app.run_server to app.run