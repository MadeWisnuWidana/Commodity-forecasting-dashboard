import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import statsmodels.api as sm
from prophet.plot import plot_components_plotly
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# =====================================================================================
# KONFIGURASI HALAMAN & DATA HARI LIBUR
# =====================================================================================
st.set_page_config(
    page_title="Dasbor Prediksi Harga",
    page_icon="📈",
    layout="wide"
)

HOLIDAYS_DATA = {
    'holiday': [
        'Tahun Baru', 'Isra Miraj', 'Tahun Baru Imlek', 'Hari Suci Nyepi', 'Wafat Isa Almasih', 'Idul Fitri', 
        'Idul Fitri', 'Hari Buruh', 'Kenaikan Isa Almasih', 'Hari Waisak', 'Hari Lahir Pancasila', 'Idul Adha', 
        'Tahun Baru Islam', 'Hari Kemerdekaan', 'Maulid Nabi Muhammad', 'Hari Natal', 'Tahun Baru', 'Isra Miraj', 
        'Tahun Baru Imlek', 'Hari Suci Nyepi', 'Idul Fitri', 'Idul Fitri', 'Wafat Isa Almasih', 'Hari Buruh', 
        'Hari Waisak', 'Kenaikan Isa Almasih', 'Hari Lahir Pancasila', 'Idul Adha', 'Tahun Baru Islam', 
        'Hari Kemerdekaan', 'Maulid Nabi Muhammad', 'Hari Natal'
    ],
    'ds': pd.to_datetime([
        '2024-01-01', '2024-02-08', '2024-02-10', '2024-03-11', '2024-03-29', '2024-04-10', '2024-04-11', 
        '2024-05-01', '2024-05-09', '2024-05-23', '2024-06-01', '2024-06-17', '2024-07-07', '2024-08-17', 
        '2024-09-16', '2024-12-25', '2025-01-01', '2025-01-27', '2025-01-29', '2025-03-29', '2025-03-31', 
        '2025-04-01', '2025-04-18', '2025-05-01', '2025-05-12', '2025-05-29', '2025-06-01', '2025-06-07', 
        '2025-06-27', '2025-08-17', '2025-09-05', '2025-12-25'
    ]),
    'lower_window': -1, 'upper_window': 1
}
HOLIDAYS_DF = pd.DataFrame(HOLIDAYS_DATA)

# =====================================================================================
# FUNGSI-FUNGSI PEMROSESAN DATA DAN MODEL
# =====================================================================================

@st.cache_data
def load_data(filepath):
    """Memuat data dari CSV dan mengonversi kolom Tanggal."""
    try:
        data = pd.read_csv(filepath)
        try:
            data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d/%m/%Y')
        except (ValueError, KeyError):
            data['Tanggal'] = pd.to_datetime(data['Tanggal'], errors='coerce')
        
        data.dropna(subset=['Tanggal'], inplace=True)
        return data
    except FileNotFoundError:
        st.error(f"File tidak ditemukan: {filepath}. Pastikan file Anda ada di folder yang sama.")
        return None

def prepare_timeseries_data(df, price_col):
    """Menyiapkan DataFrame time series 2 kolom (Date, Harga)."""
    try:
        df_ts = df[['Tanggal', price_col]].copy()
        df_ts.rename(columns={'Tanggal': 'Date', price_col: 'Harga'}, inplace=True)
        df_ts['Harga'] = pd.to_numeric(df_ts['Harga'], errors='coerce')
        df_ts.dropna(inplace=True)
        df_ts.sort_values('Date', inplace=True)
        return df_ts
    except Exception as e:
        st.error(f"Gagal memproses kolom {price_col}. Error: {e}")
        return None

def run_prophet_forecasting(data, commodity_name, forecast_period=14):
    """Menjalankan prediksi Prophet dan membuat visualisasi."""
    df_prophet = data.rename(columns={'Date': 'ds', 'Harga': 'y'})
    model = Prophet(holidays=HOLIDAYS_DF, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, holidays_prior_scale=10.0, seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_period, freq='B')
    forecast = model.predict(future)
    
    performance = pd.merge(df_prophet, forecast[['ds', 'yhat']], on='ds')
    y_true, y_pred_hist = performance['y'], performance['yhat']
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred_hist)),
        'mae': mean_absolute_error(y_true, y_pred_hist),
        'mape': np.mean(np.abs((y_true - y_pred_hist) / y_true)) * 100 if np.all(y_true != 0) else 0
    }
    
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(255, 127, 14, 0.3)', showlegend=False))
    fig_main.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255, 127, 14, 0.3)', name='Area Prediksi', hoverinfo='none'))
    fig_main.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Data Historis', line=dict(color='#1f77b4'), hovertemplate='<b>Historis</b><br>Tanggal: %{x|%d %B %Y}<br>Harga: Rp %{y:,.0f}<extra></extra>'))
    fig_main.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prediksi', line=dict(color='#ff7f0e', dash='dash'), hovertemplate='<b>Prediksi</b><br>Tanggal: %{x|%d %B %Y}<br>Harga: Rp %{y:,.0f}<extra></extra>'))
    fig_main.update_layout(title=f'Prediksi Harga {commodity_name} dengan Prophet', title_font_size=20, xaxis_title='Tanggal', yaxis_title='Harga (Rp)', yaxis_tickprefix='Rp ', yaxis_tickformat=',.0f', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), template='plotly_white', hovermode='x unified')

    fig_components = plot_components_plotly(model, forecast)
    
    forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period)
    forecast_table.rename(columns={'ds': 'Tanggal', 'yhat': 'Prediksi Harga (Rp)', 'yhat_lower': 'Batas Bawah (Rp)', 'yhat_upper': 'Batas Atas (Rp)'}, inplace=True)
    
    return fig_main, fig_components, forecast_table, metrics

def run_xgboost_forecasting(data, commodity_name, forecast_period=14):
    """Menjalankan prediksi XGBoost dan membuat visualisasi."""
    df = data.copy().set_index('Date')
    df['dayofweek'], df['quarter'], df['month'], df['year'], df['dayofyear'] = df.index.dayofweek, df.index.quarter, df.index.month, df.index.year, df.index.dayofyear
    X, y = df[['dayofweek', 'quarter', 'month', 'year', 'dayofyear']], df['Harga']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # `early_stopping_rounds` dipindahkan ke dalam constructor XGBRegressor
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=1000,
        early_stopping_rounds=50  # Parameter dipindahkan ke sini
    )
    # `early_stopping_rounds` dihapus dari .fit()
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    # --- AKHIR PERBAIKAN ---

    y_pred_test = model.predict(X_test)
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'mae': mean_absolute_error(y_test, y_pred_test),
        'mape': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100 if np.all(y_test != 0) else 0
    }
    
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period)
    future_df = pd.DataFrame(index=future_dates)
    future_df['dayofweek'], future_df['quarter'], future_df['month'], future_df['year'], future_df['dayofyear'] = future_df.index.dayofweek, future_df.index.quarter, future_df.index.month, future_df.index.year, future_df.index.dayofyear
    future_preds = model.predict(future_df)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Harga'], mode='lines', name='Data Historis', line=dict(color='#1f77b4'), hovertemplate='<b>Historis</b><br>Tanggal: %{x|%d %B %Y}<br>Harga: Rp %{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Prediksi', line=dict(color='#ff7f0e', dash='dash'), hovertemplate='<b>Prediksi</b><br>Tanggal: %{x|%d %B %Y}<br>Harga: Rp %{y:,.0f}<extra></extra>'))
    fig.update_layout(title=f'Prediksi Harga {commodity_name} dengan XGBoost', title_font_size=20, xaxis_title='Tanggal', yaxis_title='Harga (Rp)', yaxis_tickprefix='Rp ', yaxis_tickformat=',.0f', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), template='plotly_white', hovermode='x unified')

    forecast_table = pd.DataFrame({'Tanggal': future_dates, 'Prediksi Harga (Rp)': future_preds})
    return fig, None, forecast_table, metrics

def run_sarima_forecasting(data, commodity_name, forecast_period=14):
    """Menjalankan prediksi SARIMA dan membuat visualisasi."""
    df = data.copy().set_index('Date')
    y = df['Harga']

    # Otomatis mencari parameter terbaik SARIMA menggunakan AIC
    seasonal_order = (1, 1, 1, 7)  # Sesuaikan dengan siklus mingguan
    try:
        model = sm.tsa.statespace.SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
    except Exception as e:
        st.error(f"Gagal menjalankan SARIMA: {e}")
        return None, None, pd.DataFrame(), {}

    forecast = results.get_forecast(steps=forecast_period)
    forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='D')
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Akurasi berdasarkan fitted values (training data)
    y_pred_hist = results.fittedvalues
    y_true_hist = y

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true_hist, y_pred_hist)),
        'mae': mean_absolute_error(y_true_hist, y_pred_hist),
        'mape': np.mean(np.abs((y_true_hist - y_pred_hist) / y_true_hist)) * 100
                if np.all(y_true_hist != 0) else 0
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y, mode='lines', name='Data Historis', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines', name='Prediksi', line=dict(color='#ff7f0e', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 0], fill=None, mode='lines', line_color='rgba(255,127,14,0.3)', showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 1], fill='tonexty', mode='lines', line_color='rgba(255,127,14,0.3)', name='Area Prediksi'))
    fig.update_layout(title=f'Prediksi Harga {commodity_name} dengan SARIMA', xaxis_title='Tanggal', yaxis_title='Harga (Rp)', yaxis_tickprefix='Rp ', yaxis_tickformat=',.0f', template='plotly_white')

    forecast_table = pd.DataFrame({
        'Tanggal': forecast_index,
        'Prediksi Harga (Rp)': forecast_mean,
        'Batas Bawah (Rp)': conf_int.iloc[:, 0],
        'Batas Atas (Rp)': conf_int.iloc[:, 1]
    })
    return fig, None, forecast_table, metrics

@st.cache_data(show_spinner=False)
def run_cached_forecasting(model_name, data, commodity):
    """Fungsi pembungkus untuk caching hasil model."""
    if model_name == "Prophet":
        return run_prophet_forecasting(data, commodity)
    elif model_name == "XGBoost":
        return run_xgboost_forecasting(data, commodity)
    elif model_name == "SARIMA":
        return run_sarima_forecasting(data, commodity)


# =====================================================================================
# TATA LETAK APLIKASI STREAMLIT
# =====================================================================================
def main():
    """Fungsi utama untuk menjalankan aplikasi Streamlit."""
    st.title("📈 Dasbor Prediksi Harga Komoditas Pokok")
    st.write("Selamat datang di dasbor interaktif untuk memprediksi harga komoditas kebutuhan pokok di Indonesia.")
    st.markdown("---")

    data_raw = load_data('Harga_Nasional_Pivot.csv')
    
    if data_raw is not None:
        # --- SIDEBAR UNTUK INPUT PENGGUNA ---
        with st.sidebar:
            st.header("⚙️ Pengaturan Prediksi")
            commodities = [col for col in data_raw.columns if col != 'Tanggal']
            selected_commodity = st.selectbox("1. Pilih Komoditas", options=commodities)
            selected_method = st.selectbox("2. Pilih Metode Prediksi", options=["Prophet", "XGBoost", "SARIMA"])
            process_button = st.button("🚀 Proses Prediksi", type="primary", use_container_width=True)

        # --- AREA TAMPILAN UTAMA ---
        st.header("Data Historis Harga Komoditas Nasional Rp.")
        # Menentukan format untuk kolom numerik agar tampil 3 angka di belakang koma
        format_dict = {"Tanggal": "{:%d-%m-%Y}"}
        for col in data_raw.columns:
            if col != "Tanggal":
                format_dict[col] = "{:.3f}"
        # Tampilkan tabel dengan format yang diperbarui
        st.dataframe(data_raw.style.format(format_dict), use_container_width=True)
        st.info("Anda dapat mengurutkan data dengan mengklik pada header kolom.")
        st.markdown("---")
        
        st.header("📊 Hasil Prediksi Harga")
        
        if process_button:
            timeseries_data = prepare_timeseries_data(data_raw, selected_commodity)
            if timeseries_data is None or timeseries_data.empty or len(timeseries_data) < 30:
                st.warning(f"Tidak cukup data valid untuk '{selected_commodity}' untuk membuat prediksi.")
            else:
                with st.spinner(f"Memproses prediksi untuk **{selected_commodity}**... (Proses pertama mungkin memakan waktu)"):
                    fig, fig_komponen, forecast_df, metrics = run_cached_forecasting(selected_method, timeseries_data, selected_commodity)
                
                st.success("Prediksi berhasil dibuat!")
                
                st.subheader("Tingkat Akurasi Model")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("RMSE", f"{metrics['rmse']:,.2f}")
                m_col2.metric("MAE", f"{metrics['mae']:,.2f}")
                m_col3.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
                
                st.subheader("Grafik Prediksi Harga")
                st.plotly_chart(fig, use_container_width=True)

                if fig_komponen:
                    st.subheader("Grafik Komponen Prediksi (Tren & Musiman)")
                    st.plotly_chart(fig_komponen, use_container_width=True)

                st.subheader("Tabel Prediksi Harga untuk 14 Hari ke Depan")
                forecast_df['Tanggal'] = pd.to_datetime(forecast_df['Tanggal']).dt.strftime('%d-%m-%Y')
                for col in ['Prediksi Harga (Rp)', 'Batas Bawah (Rp)', 'Batas Atas (Rp)']:
                    if col in forecast_df.columns:
                        forecast_df[col] = forecast_df[col].apply(lambda x: f"Rp {x:,.2f}")
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        else:
            st.info("Silakan pilih komoditas dan metode di panel sebelah kiri, lalu klik **'Proses Prediksi'**.")
    else:
        st.warning("Gagal memuat data. Silakan periksa file 'Harga_Nasional_Pivot.csv'.")

if __name__ == "__main__":
    main()