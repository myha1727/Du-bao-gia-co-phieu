import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
from datetime import date, timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# TensorFlow is used for the LSTM model from the source notebook.
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# =========================================================
# PAGE CONFIG + STYLE
# =========================================================
st.set_page_config(
    page_title="AI Stock Forecast",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: #155EEF;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        color: #667085;
        font-size: 1rem;
        margin-bottom: 1.25rem;
    }
    div[data-testid="stMetric"] {
        background: #F5F8FF;
        border: 1px solid #E0E7FF;
        padding: 14px;
        border-radius: 12px;
    }
    .note {
        background: #FFF8E7;
        border-left: 4px solid #F5B700;
        padding: 12px 14px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 AI Dự báo giá cổ phiếu</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Mô hình LSTM học từ dữ liệu giá đóng cửa lịch sử để thử dự báo 10 phiên giao dịch tiếp theo.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="note">Lưu ý: Đây là ứng dụng học tập/minh họa, không phải công cụ khuyến nghị đầu tư. '
    'Dự báo có thể sai lệch mạnh và không đảm bảo giá sẽ diễn biến như mô hình dự đoán.</div>',
    unsafe_allow_html=True
)


# =========================================================
# HELPERS
# =========================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Tải và chuẩn hóa dữ liệu giá đóng cửa từ Yahoo Finance."""
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=30,
            multi_level_index=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # Chuẩn hóa tên cột, kể cả khi Yahoo trả về MultiIndex.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(col).strip() for col in df.columns]

        if "Close" not in df.columns:
            return pd.DataFrame()

        result = df[["Close"]].copy()
        result["Close"] = pd.to_numeric(
            result["Close"], errors="coerce"
        )
        result = result.dropna()
        result = result[~result.index.duplicated(keep="last")]
        result = result.sort_index()
        result.index = pd.to_datetime(result.index)

        return result

    except Exception as e:
        st.warning(f"Không thể tải dữ liệu từ Yahoo Finance: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Newer yfinance versions may return MultiIndex columns even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    if "Close" not in df.columns:
        return pd.DataFrame()

    result = df[["Close"]].copy()
    result["Close"] = pd.to_numeric(result["Close"], errors="coerce")
    result = result.dropna()
    result.index = pd.to_datetime(result.index)
    return result


def build_sequences(values: np.ndarray, window_size: int = 60):
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i, 0])
        y.append(values[i, 0])
    return np.asarray(X), np.asarray(y)


def make_lstm_model(window_size: int = 60):
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def get_next_business_days(last_day, count=10):
    """Generate future weekdays, excluding Saturdays and Sundays."""
    days = []
    current = pd.Timestamp(last_day).date()
    while len(days) < count:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days.append(current)
    return pd.to_datetime(days)


def money_format(value):
    return f"{value:,.2f}"


def ticker_from_input(user_ticker: str) -> str:
    """
    Yahoo Finance commonly uses .VN for Vietnamese tickers.
    If user enters VNM, use VNM.VN; if user enters VNM.VN, keep it.
    """
    ticker = user_ticker.strip().upper().replace(" ", "")
    if not ticker:
        return ""
    if "." not in ticker and ticker.isalpha():
        ticker += ".VN"
    return ticker


# =========================================================
# SIDEBAR INPUTS
# =========================================================
with st.sidebar:
    st.header("⚙️ Thiết lập mô hình")
    raw_ticker = st.text_input(
        "Mã cổ phiếu",
        value="VNM",
        help="Ví dụ cổ phiếu Việt Nam: VNM, FPT, HPG, MWG. Có thể nhập đầy đủ mã Yahoo Finance như VNM.VN."
    )
    years_history = st.selectbox(
        "Dữ liệu lịch sử",
        options=[3, 5, 7, 8],
        index=1,
        format_func=lambda x: f"{x} năm gần nhất"
    )
    epochs = st.slider("Số epochs huấn luyện", min_value=5, max_value=30, value=10, step=5)
    run_button = st.button("🚀 Phân tích và dự báo", type="primary", use_container_width=True)

    st.caption("Mô hình dùng giá đóng cửa, cửa sổ 60 phiên và chia dữ liệu theo thứ tự thời gian.")


if not TF_AVAILABLE:
    st.error(
        "Chưa cài được TensorFlow. Hãy cài các thư viện theo hướng dẫn ở cuối ứng dụng, "
        "sau đó chạy lại Streamlit."
    )
    st.stop()

if not run_button:
    st.info("Nhập mã cổ phiếu ở thanh bên trái rồi nhấn **Phân tích và dự báo**.")
    st.markdown("### Ứng dụng sẽ hiển thị")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- Đánh giá mô hình bằng RMSE, MAE và R²")
        st.markdown("- Dự báo giá cho 10 phiên giao dịch tiếp theo")
    with col2:
        st.markdown("- Biểu đồ giá thật so với giá mô hình dự báo")
        st.markdown("- Nhận định xu hướng và gợi ý mua/bán giả lập")
    st.stop()


ticker = ticker_from_input(raw_ticker)
if not ticker:
    st.error("Vui lòng nhập mã cổ phiếu.")
    st.stop()

end_date = date.today() + timedelta(days=1)
start_date = end_date - timedelta(days=365 * years_history)

try:
    with st.spinner(f"Đang tải dữ liệu {ticker} và huấn luyện mô hình LSTM..."):
        df = load_stock_data(
            ticker,
            start_date.isoformat(),
            end_date.isoformat()
        )

        if df.empty and not ticker.endswith(".VN"):
            st.warning("Không tìm thấy dữ liệu với mã đã nhập. Hãy kiểm tra lại mã cổ phiếu.")
            st.stop()
        elif df.empty:
            st.error(
                f"Không tải được dữ liệu cho {ticker}. "
                "Yahoo Finance có thể không hỗ trợ mã này hoặc dữ liệu hiện không khả dụng. "
                "Thử nhập mã Yahoo Finance đầy đủ, ví dụ VNM.VN."
            )
            st.stop()

        if len(df) < 120:
            st.error(
                f"Chỉ có {len(df)} phiên dữ liệu. Cần tối thiểu khoảng 120 phiên để huấn luyện mô hình."
            )
            st.stop()

        close_values = df[["Close"]].values.astype(float)
        window_size = 60

        # Chronological split: no random shuffle.
        train_size = int(len(close_values) * 0.8)
        if train_size <= window_size or len(close_values) - train_size < 10:
            st.error("Dữ liệu chưa đủ để chia tập huấn luyện và kiểm tra.")
            st.stop()

        # Fit scaler only on training data to reduce data leakage.
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(close_values[:train_size])
        all_scaled = scaler.transform(close_values)

        X_train, y_train = build_sequences(train_scaled, window_size)
        test_scaled = all_scaled[train_size - window_size:]
        X_test, y_test = build_sequences(test_scaled, window_size)

        if len(X_train) < 10 or len(X_test) < 5:
            st.error("Không đủ dữ liệu sau khi tạo chuỗi. Hãy chọn khoảng lịch sử dài hơn.")
            st.stop()

        X_train = X_train.reshape((X_train.shape[0], window_size, 1))
        X_test = X_test.reshape((X_test.shape[0], window_size, 1))

        # Last 10% of training sequences are used as validation data.
        val_size = max(1, int(len(X_train) * 0.1))
        X_train_final, X_val = X_train[:-val_size], X_train[-val_size:]
        y_train_final, y_val = y_train[:-val_size], y_train[-val_size:]

        if len(X_train_final) == 0:
            st.error("Không đủ chuỗi để huấn luyện. Hãy chọn dữ liệu lịch sử dài hơn.")
            st.stop()

        # Reproducible initialization as in the supplied source notebook.
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

        model = make_lstm_model(window_size)
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        )

        history = model.fit(
            X_train_final,
            y_train_final,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

        test_pred_scaled = model.predict(X_test, verbose=0)
        test_pred = scaler.inverse_transform(test_pred_scaled).flatten()
        y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        test_dates = df.index[train_size:]

        rmse = float(np.sqrt(mean_squared_error(y_test_real, test_pred)))
        mae = float(mean_absolute_error(y_test_real, test_pred))
        r2 = float(r2_score(y_test_real, test_pred)) if len(y_test_real) > 1 else float("nan")

        # Recursive 10-step forecast based on the most recent 60 observed prices.
        last_window = all_scaled[-window_size:].reshape(1, window_size, 1)
        future_scaled = []
        for _ in range(10):
            next_scaled = model.predict(last_window, verbose=0)[0, 0]
            future_scaled.append(next_scaled)
            next_item = np.array(next_scaled).reshape(1, 1, 1)
            last_window = np.concatenate([last_window[:, 1:, :], next_item], axis=1)

        future_prices = scaler.inverse_transform(
            np.array(future_scaled).reshape(-1, 1)
        ).flatten()
        future_dates = get_next_business_days(df.index[-1], 10)

except Exception as e:
    st.error("Ứng dụng gặp lỗi khi tải dữ liệu hoặc huấn luyện mô hình.")
    st.exception(e)
    st.stop()


# =========================================================
# RESULTS
# =========================================================
latest_price = float(close_values[-1, 0])
first_future = float(future_prices[0])
last_future = float(future_prices[-1])
change_pct = ((last_future - latest_price) / latest_price * 100) if latest_price else 0
predicted_up_days = int(np.sum(np.diff(np.r_[latest_price, future_prices]) > 0))
predicted_down_days = int(np.sum(np.diff(np.r_[latest_price, future_prices]) < 0))

st.subheader(f"📌 Kết quả phân tích: {ticker}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Giá đóng cửa gần nhất", money_format(latest_price))
m2.metric("Dự báo phiên thứ 10", money_format(last_future), f"{change_pct:+.2f}% so với giá gần nhất")
m3.metric("RMSE (tập kiểm tra)", money_format(rmse))
m4.metric("MAE (tập kiểm tra)", money_format(mae))

st.caption(
    f"Dữ liệu từ {df.index.min().date()} đến {df.index.max().date()} · "
    f"{len(df)} phiên giao dịch · Giá dự báo và giá lịch sử giữ nguyên đơn vị do nguồn dữ liệu cung cấp."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Dự báo 10 phiên",
    "📉 Giá thật và dự báo",
    "🧪 Đánh giá mô hình",
    "🤖 Gợi ý mua/bán giả lập"
])

with tab1:
    forecast_df = pd.DataFrame({
        "Phiên dự kiến": range(1, 11),
        "Ngày dự kiến": future_dates.date,
        "Giá dự báo": future_prices
    })
    st.dataframe(
        forecast_df.style.format({"Giá dự báo": "{:,.2f}"}),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(future_dates, future_prices, marker="o", linewidth=2, label="Giá dự báo")
    ax.axhline(latest_price, linestyle="--", linewidth=1.5, label="Giá đóng cửa gần nhất")
    ax.set_title(f"Dự báo giá {ticker} trong 10 phiên giao dịch tiếp theo")
    ax.set_xlabel("Ngày dự kiến")
    ax.set_ylabel("Giá đóng cửa")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if change_pct > 1:
        st.success(f"Mô hình dự báo xu hướng tăng khoảng {change_pct:.2f}% sau 10 phiên so với giá đóng cửa gần nhất.")
    elif change_pct < -1:
        st.warning(f"Mô hình dự báo xu hướng giảm khoảng {abs(change_pct):.2f}% sau 10 phiên so với giá đóng cửa gần nhất.")
    else:
        st.info(f"Mô hình dự báo giá sau 10 phiên tương đối đi ngang ({change_pct:+.2f}%).")

with tab2:
    actual_series = pd.Series(y_test_real, index=test_dates[:len(y_test_real)])
    pred_series = pd.Series(test_pred, index=test_dates[:len(test_pred)])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual_series.index, actual_series.values, label="Giá thật", linewidth=1.8)
    ax.plot(pred_series.index, pred_series.values, label="Giá dự báo của mô hình", linewidth=1.5)
    ax.set_title(f"Giá thật và giá dự báo trên tập kiểm tra — {ticker}")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá đóng cửa")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption("Đây là phần dữ liệu kiểm tra mà mô hình chưa dùng để cập nhật trọng số trong quá trình huấn luyện.")

with tab3:
    st.markdown("### Các chỉ số đánh giá")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", money_format(rmse), help="Sai số bình phương trung bình căn bậc hai; càng thấp thường càng tốt.")
    c2.metric("MAE", money_format(mae), help="Sai số tuyệt đối trung bình; càng thấp thường càng tốt.")
    c3.metric("R²", f"{r2:.4f}", help="Độ phù hợp trên tập kiểm tra; có thể âm nếu dự báo kém hơn mốc trung bình.")

    st.markdown("""
    - **RMSE / MAE:** đo mức chênh lệch giữa giá thật và giá dự báo, theo đơn vị giá của dữ liệu.
    - **R²:** phản ánh mức độ mô hình giải thích biến động trong tập kiểm tra; R² không bảo đảm mô hình dự báo tốt trong tương lai.
    - Không có một ngưỡng duy nhất để kết luận mô hình tốt cho mọi mã cổ phiếu. Nên so sánh với mô hình cơ sở, chẳng hạn dự báo giá ngày mai bằng giá hôm nay.
    """)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history.get("loss", []), label="Train loss")
    ax.plot(history.history.get("val_loss", []), label="Validation loss")
    ax.set_title("Loss khi huấn luyện mô hình LSTM")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error (scaled)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab4:
    st.markdown("### 🤖 Nhận định tự động (mô phỏng)")
    st.write(
        f"So sánh giá đóng cửa gần nhất (**{money_format(latest_price)}**) với giá dự báo ở phiên thứ 10 "
        f"(**{money_format(last_future)}**), mô hình cho mức thay đổi dự kiến **{change_pct:+.2f}%**."
    )

    if change_pct >= 3 and r2 > 0:
        st.success("Kịch bản mô phỏng: **NGHIÊNG VỀ MUA / THEO DÕI ĐIỂM MUA**")
        st.write("Điều kiện giả lập: mức tăng dự báo tương đối rõ và R² trên tập kiểm tra dương.")
    elif change_pct <= -3 and r2 > 0:
        st.error("Kịch bản mô phỏng: **NGHIÊNG VỀ BÁN / GIẢM TỶ TRỌNG**")
        st.write("Điều kiện giả lập: mức giảm dự báo tương đối rõ và R² trên tập kiểm tra dương.")
    else:
        st.info("Kịch bản mô phỏng: **CHỜ TÍN HIỆU / CHƯA CÓ KẾT LUẬN RÕ**")
        st.write("Biến động dự báo chưa đủ mạnh theo ngưỡng giả lập, hoặc chỉ số đánh giá chưa hỗ trợ nhận định rõ.")

    st.markdown("**Đếm hướng biến động trong 10 bước dự báo:**")
    st.write(f"- Số bước tăng: {predicted_up_days}")
    st.write(f"- Số bước giảm: {predicted_down_days}")

    st.warning(
        "Gợi ý mua/bán ở đây chỉ là quy tắc minh họa dựa trên dự báo giá và R², "
        "không phải lời khuyên đầu tư. Mô hình chỉ dùng giá đóng cửa, chưa xét khối lượng, "
        "tin tức, tình hình doanh nghiệp, phí giao dịch, thanh khoản hay khẩu vị rủi ro."
    )

st.divider()
st.markdown("### Cách cài đặt và chạy")
st.code(
    "pip install streamlit yfinance pandas numpy matplotlib scikit-learn tensorflow\n"
    "streamlit run app.py",
    language="bash"
)
st.caption(
    "Lưu ý dữ liệu: Yahoo Finance có thể thiếu hoặc chậm dữ liệu đối với một số mã trên thị trường Việt Nam. "
    "Nếu mã không tải được, hãy thử dạng mã đầy đủ trên Yahoo Finance (ví dụ VNM.VN) hoặc kiểm tra lại nguồn dữ liệu."
)
