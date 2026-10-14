
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# =====================================================
# 1. CẤU HÌNH ỨNG DỤNG
# =====================================================
st.set_page_config(
    page_title="AI Stock Forecast",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
[data-testid="stMetric"] {
    background: white;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #e5eaf2;
}
h1, h2, h3 {
    color: #15345b;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 AI STOCK FORECAST")
st.caption(
    "Phân tích dữ liệu thị trường và dự báo giá cổ phiếu "
    "bằng Machine Learning"
)

# =====================================================
# 2. TẢI DỮ LIỆU
# =====================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_stock_data(ticker, period):
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    # Xử lý trường hợp yfinance trả về MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Close", "Volume"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    df = df[required].copy()
    df = df.dropna(subset=["Close"])
    df["Volume"] = df["Volume"].fillna(0)
    df = df[~df.index.duplicated(keep="last")]

    return df


# =====================================================
# 3. TẠO ĐẶC TRƯNG CHO AI
# =====================================================
LAGS = 20


def make_features(history):
    """Tạo đặc trưng chỉ từ giá đã quan sát."""
    x = np.asarray(history, dtype=float)

    if len(x) < LAGS:
        raise ValueError("Không đủ dữ liệu để tạo đặc trưng.")

    features = {}

    # Giá đóng cửa các phiên trước
    for lag in range(1, LAGS + 1):
        features[f"close_lag_{lag}"] = x[-lag]

    # Trung bình động
    for window in [5, 10, 20]:
        features[f"sma_{window}"] = np.mean(x[-window:])

    # Biến động giá và động lượng
    features["return_1"] = x[-1] / x[-2] - 1
    features["return_5"] = x[-1] / x[-6] - 1
    features["volatility_10"] = np.std(
        np.diff(np.log(x[-11:]))
    )

    return features


def make_training_data(close_prices):
    """Dùng dữ liệu trước phiên i để dự báo giá phiên i."""
    x_rows = []
    y_rows = []

    for i in range(LAGS, len(close_prices)):
        x_rows.append(
            make_features(close_prices[:i])
        )
        y_rows.append(close_prices[i])

    return pd.DataFrame(x_rows), np.asarray(y_rows)


def create_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    )


# =====================================================
# 4. GIAO DIỆN ĐIỀU KHIỂN
# =====================================================
with st.sidebar:
    st.header("⚙️ Cấu hình dự báo")

    ticker = st.text_input(
        "Mã cổ phiếu Yahoo Finance",
        value="AAPL",
        help=(
            "Ví dụ: AAPL, MSFT, NVDA. "
            "Một số mã Việt Nam có hậu tố .VN, "
            "như FPT.VN nếu được nguồn dữ liệu hỗ trợ."
        )
    ).strip().upper()

    period = st.selectbox(
        "Dữ liệu lịch sử",
        ["1y", "2y", "5y", "10y"],
        index=1
    )

    forecast_days = st.slider(
        "Số phiên muốn dự báo",
        min_value=1,
        max_value=30,
        value=10
    )

    run = st.button(
        "🚀 Phân tích và dự báo",
        type="primary",
        use_container_width=True
    )

    st.caption(
        "Nguồn dữ liệu: Yahoo Finance thông qua yfinance."
    )

if not ticker:
    st.warning("Vui lòng nhập mã cổ phiếu.")
    st.stop()

# Tự chạy khi mở ứng dụng; nút dùng để tải lại dữ liệu
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ticker

if run:
    st.cache_data.clear()
    st.session_state.last_ticker = ticker

try:
    with st.spinner("Đang tải dữ liệu thị trường..."):
        df = load_stock_data(
            ticker,
            period
        )
except Exception as exc:
    st.error(f"Không tải được dữ liệu: {exc}")
    st.stop()

if df.empty or len(df) < 100:
    st.error(
        "Không có đủ dữ liệu lịch sử. "
        "Hãy kiểm tra mã chứng khoán hoặc chọn mã khác."
    )
    st.stop()

close_prices = df["Close"].astype(float).values
current_price = float(close_prices[-1])

# =====================================================
# 5. TỔNG QUAN CỔ PHIẾU
# =====================================================
previous_price = float(close_prices[-2])
change = current_price - previous_price
change_pct = change / previous_price * 100 if previous_price else 0

sma20 = float(pd.Series(close_prices).tail(20).mean())
sma50 = float(pd.Series(close_prices).tail(50).mean())

st.subheader(f"📊 Tổng quan: {ticker}")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Giá đóng cửa gần nhất",
    f"{current_price:,.2f}",
    f"{change:+,.2f} ({change_pct:+.2f}%)"
)

c2.metric(
    "Trung bình động 20 phiên",
    f"{sma20:,.2f}"
)

c3.metric(
    "Trung bình động 50 phiên",
    f"{sma50:,.2f}"
)

c4.metric(
    "Số phiên dữ liệu",
    f"{len(df):,}"
)

# =====================================================
# 6. BIỂU ĐỒ GIÁ LỊCH SỬ
# =====================================================
st.subheader("📉 Diễn biến giá cổ phiếu")

fig_history = go.Figure()

fig_history.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Giá đóng cửa"
    )
)

fig_history.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"].rolling(20).mean(),
        mode="lines",
        name="SMA 20"
    )
)

fig_history.update_layout(
    template="plotly_white",
    height=430,
    xaxis_title="Ngày",
    yaxis_title="Giá",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.1)
)

st.plotly_chart(
    fig_history,
    use_container_width=True
)

# =====================================================
# 7. HUẤN LUYỆN VÀ KIỂM TRA MÔ HÌNH
# =====================================================
st.subheader("🤖 Dự báo bằng Random Forest")

X, y = make_training_data(close_prices)

# Chia theo thời gian, không xáo trộn dữ liệu
split = int(len(X) * 0.8)

if split < 30 or len(X) - split < 10:
    st.error("Dữ liệu chưa đủ để đánh giá mô hình.")
    st.stop()

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y[:split], y[split:]

with st.spinner("Đang huấn luyện mô hình AI..."):
    evaluation_model = create_model()
    evaluation_model.fit(X_train, y_train)

    test_predictions = evaluation_model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

# Mốc tham chiếu: dự báo giá phiên tiếp theo bằng giá phiên trước
baseline_predictions = X_test["close_lag_1"].values
baseline_mae = mean_absolute_error(y_test, baseline_predictions)

m1, m2, m3 = st.columns(3)

m1.metric("MAE của AI", f"{mae:,.3f}")
m2.metric("RMSE của AI", f"{rmse:,.3f}")
m3.metric("MAE mốc tham chiếu", f"{baseline_mae:,.3f}")

st.caption(
    "MAE và RMSE càng thấp thường càng tốt trên cùng tập kiểm tra. "
    "Hãy so sánh với mốc tham chiếu để biết mô hình có thực sự hữu ích."
)

# Biểu đồ kiểm tra
test_dates = df.index[LAGS + split:]

fig_test = go.Figure()

fig_test.add_trace(
    go.Scatter(
        x=test_dates,
        y=y_test,
        mode="lines",
        name="Giá thực tế"
    )
)

fig_test.add_trace(
    go.Scatter(
        x=test_dates,
        y=test_predictions,
        mode="lines",
        name="Giá AI dự báo"
    )
)

fig_test.update_layout(
    template="plotly_white",
    height=400,
    title="Giá thực tế và giá dự báo trên tập kiểm tra",
    xaxis_title="Ngày",
    yaxis_title="Giá"
)

st.plotly_chart(
    fig_test,
    use_container_width=True
)

# =====================================================
# 8. DỰ BÁO CÁC PHIÊN TIẾP THEO
# =====================================================
# Huấn luyện lại bằng toàn bộ dữ liệu có sẵn
final_model = create_model()
final_model.fit(X, y)

history = list(close_prices)
future_prices = []

for _ in range(forecast_days):
    next_features = pd.DataFrame(
        [make_features(history)]
    )

    next_price = float(
        final_model.predict(next_features)[0]
    )

    future_prices.append(next_price)
    history.append(next_price)

# Ngày làm việc dự kiến; không loại trừ ngày nghỉ lễ
future_dates = pd.bdate_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days
)

forecast_df = pd.DataFrame({
    "Ngày dự kiến": future_dates,
    "Giá dự báo": future_prices
})

st.subheader(
    f"🔮 Dự báo {forecast_days} phiên tiếp theo"
)

last_forecast = future_prices[-1]
pred_change = last_forecast - current_price
pred_change_pct = (
    pred_change / current_price * 100
    if current_price else 0
)

f1, f2, f3 = st.columns(3)

f1.metric(
    "Giá dự báo cuối kỳ",
    f"{last_forecast:,.2f}"
)

f2.metric(
    "Chênh lệch so với hiện tại",
    f"{pred_change:+,.2f}"
)

f3.metric(
    "Biến động mô hình dự báo",
    f"{pred_change_pct:+.2f}%"
)

# Ghép giá lịch sử và giá dự báo
fig_forecast = go.Figure()

fig_forecast.add_trace(
    go.Scatter(
        x=df.index[-90:],
        y=df["Close"].tail(90),
        mode="lines",
        name="Giá lịch sử"
    )
)

fig_forecast.add_trace(
    go.Scatter(
        x=[df.index[-1]] + list(future_dates),
        y=[current_price] + future_prices,
        mode="lines+markers",
        name="Đường dự báo AI",
        line=dict(dash="dash")
    )
)

fig_forecast.update_layout(
    template="plotly_white",
    height=450,
    title="Kịch bản giá theo mô hình AI",
    xaxis_title="Ngày",
    yaxis_title="Giá",
    hovermode="x unified"
)

st.plotly_chart(
    fig_forecast,
    use_container_width=True
)

st.dataframe(
    forecast_df,
    use_container_width=True,
    hide_index=True
)

csv_data = forecast_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "📥 Tải kết quả dự báo CSV",
    data=csv_data,
    file_name=f"du_bao_{ticker}.csv",
    mime="text/csv"
)

# =====================================================
# 9. LƯU Ý
# =====================================================
with st.expander("ℹ️ Mô hình hoạt động như thế nào?"):
    st.markdown("""
    **Dữ liệu đầu vào**
    - Giá đóng cửa lịch sử.
    - Giá trễ 1–20 phiên.
    - Trung bình động 5, 10, 20 phiên.
    - Biến động giá và động lượng ngắn hạn.

    **Thuật toán**
    - Random Forest Regressor.
    - Chia dữ liệu theo thứ tự thời gian: 80% huấn luyện,
      20% kiểm tra.
    - Sau khi đánh giá, mô hình được huấn luyện lại trên
      toàn bộ dữ liệu để tạo dự báo tương lai.
    - Dự báo nhiều phiên sử dụng dự báo trước đó làm đầu vào
      cho bước tiếp theo.

    **Hạn chế**
    - Chưa tính tin tức, báo cáo tài chính, lãi suất,
      thanh khoản thị trường hoặc sự kiện bất thường.
    - Sai số trên dữ liệu quá khứ không bảo đảm độ chính xác
      trong tương lai.
    - Ngày dự báo là ngày làm việc dự kiến, chưa loại trừ
      toàn bộ ngày nghỉ của từng sở giao dịch.
    """)

st.warning(
    "⚠️ Đây là công cụ nghiên cứu và học tập, không phải "
    "khuyến nghị mua/bán. Giá dự báo do mô hình tạo ra "
    "không phải giá chắc chắn sẽ xảy ra."
)
