import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Smart Energy Monitoring",
    page_icon="⚡",
    layout="wide"
)

# Load data and trained model
hourly_data = pd.read_csv("hourly_energy.csv")
daily_energy = pd.read_csv("daily_energy.csv")

hourly_data["Timestamp"] = pd.to_datetime(hourly_data["Timestamp"])
daily_energy["Date"] = pd.to_datetime(daily_energy["Date"])

model = joblib.load("energy_model.pkl")

# Dashboard heading
st.title("⚡ Smart Energy Monitoring & Prediction")
st.write(
    "Monitor electricity parameters, analyze energy consumption "
    "and predict next-day energy usage."
)

# Sidebar date filter
st.sidebar.header("Filter Data")

available_dates = sorted(
    hourly_data["Timestamp"].dt.date.unique()
)

min_date = available_dates[0]
max_date = available_dates[-1]

start_date_text = st.sidebar.text_input(
    "Start date (YYYY-MM-DD)",
    value=str(min_date)
)

end_date_text = st.sidebar.text_input(
    "End date (YYYY-MM-DD)",
    value=str(max_date)
)

try:
    start_date = pd.to_datetime(
        start_date_text, format="%Y-%m-%d", errors="raise"
    ).date()
    end_date = pd.to_datetime(
        end_date_text, format="%Y-%m-%d", errors="raise"
    ).date()

    if start_date < min_date or end_date > max_date:
        st.sidebar.error(
            f"Dates must be between {min_date} and {max_date}."
        )
        st.stop()

    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

except (ValueError, TypeError):
    st.sidebar.error("Enter dates in YYYY-MM-DD format.")
    st.stop()

# Filter hourly data
filtered_data = hourly_data[
    (hourly_data["Timestamp"].dt.date >= start_date) &
    (hourly_data["Timestamp"].dt.date <= end_date)
].copy()

if filtered_data.empty:
    st.warning("No readings available for this date range.")
    st.stop()

# Monitoring metrics
st.subheader("Electrical Parameter Monitoring")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Voltage",
    f"{filtered_data['Voltage_V'].mean():.2f} V"
)
col2.metric(
    "Average Current",
    f"{filtered_data['Current_A'].mean():.3f} A"
)
col3.metric(
    "Average Power",
    f"{filtered_data['Power_W'].mean():.2f} W"
)
col4.metric(
    "Energy in Selected Period",
    f"{filtered_data['Energy_kWh'].sum():.2f} kWh"
)

# Next-day prediction
st.subheader("Next-Day Energy Prediction")

latest_3_days = daily_energy.sort_values("Date").tail(3)

forecast_input = pd.DataFrame({
    "Energy_today": [
        latest_3_days["Daily_Energy_kWh"].iloc[-1]
    ],
    "Energy_1_day_ago": [
        latest_3_days["Daily_Energy_kWh"].iloc[-2]
    ],
    "Energy_2_days_ago": [
        latest_3_days["Daily_Energy_kWh"].iloc[-3]
    ]
})

forecast_input = forecast_input[
    ["Energy_today", "Energy_1_day_ago", "Energy_2_days_ago"]
]

prediction = model.predict(forecast_input)[0]

latest_energy = latest_3_days["Daily_Energy_kWh"].iloc[-1]
latest_date = latest_3_days["Date"].iloc[-1]
forecast_date = latest_date + pd.Timedelta(days=1)

p1, p2 = st.columns(2)
p1.metric("Latest Daily Energy", f"{latest_energy:.3f} kWh")
p2.metric("Predicted Next-Day Energy", f"{prediction:.3f} kWh")

st.caption(f"Forecast date: {forecast_date.strftime('%d %b %Y')}")

# Energy consumption graph
st.subheader("Energy Consumption Over Time")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    filtered_data["Timestamp"],
    filtered_data["Energy_kWh"],
    marker=".",
    linewidth=1,
    label="Hourly Energy"
)
ax.set_xlabel("Time")
ax.set_ylabel("Energy (kWh)")
ax.grid(True)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Voltage, current and power graphs
st.subheader("Electrical Parameters Over Time")

tab1, tab2, tab3 = st.tabs(["Voltage", "Current", "Power"])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(filtered_data["Timestamp"], filtered_data["Voltage_V"])
    ax.set_ylabel("Voltage (V)")
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(filtered_data["Timestamp"], filtered_data["Current_A"])
    ax.set_ylabel("Current (A)")
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(filtered_data["Timestamp"], filtered_data["Power_W"])
    ax.set_ylabel("Power (W)")
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Historical table
st.subheader("Hourly Readings")
st.dataframe(
    filtered_data.sort_values("Timestamp", ascending=False),
    use_container_width=True
)

st.caption(
    "Demo uses simulated historical data. "
    "Predictions are not live electricity readings."
)
