import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LSTM Demand Classification",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5fff8;
}

/* All normal text */
.stApp p,
.stApp li,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6,
.stApp label {
    color: #000000 !important;
}


/* Main Title */
.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #000000 !important;
    margin-bottom: 5px;
}


/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #000000 !important;
    margin-bottom: 30px;
}


/* About Model Card */
.card {
    padding: 25px;
    border-radius: 15px;
    background-color: #ffffff;
    color: #000000 !important;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.10);
    margin-top: 20px;
}

.card h3,
.card p,
.card li {
    color: #000000 !important;
}


/* Result Card */
.result {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: #000000 !important;
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.10);
}


/* Input labels */
.stNumberInput label {
    color: #000000 !important;
}


/* Input boxes */
.stNumberInput input {
    color: #000000 !important;
    background-color: #ffffff !important;
}


/* Input text */
input {
    color: #000000 !important;
}


/* Predict Button */
.stButton button {
    background-color: #198754 !important;
    color: #ffffff !important;
    border-radius: 10px;
    border: none;
    padding: 10px;
    font-size: 18px;
    font-weight: bold;
}


/* Button hover */
.stButton button:hover {
    background-color: #146c43 !important;
    color: #ffffff !important;
}


/* Metric values */
[data-testid="stMetricValue"] {
    color: #000000 !important;
}


/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #000000 !important;
}


/* Expander / other text */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">🛒 LSTM Demand Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Store 44 – Grocery Sales Demand Prediction</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD LSTM MODEL
# ============================================================

@st.cache_resource
def load_lstm_model():

    return load_model(
        "lstm_demand_model.keras"
    )


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    with open(
        "lstm_scaler.pkl",
        "rb"
    ) as file:

        return pickle.load(file)


# ============================================================
# LOAD FILES
# ============================================================

try:

    model = load_lstm_model()

    scaler = load_scaler()

except Exception as e:

    st.error(
        "Model files could not be loaded."
    )

    st.info(
        "Make sure these files are in the same folder "
        "as app_lstm.py:"
    )

    st.code(
        "lstm_demand_model.keras\n"
        "lstm_scaler.pkl"
    )

    st.stop()


# ============================================================
# ABOUT MODEL
# ============================================================

st.markdown("""
<div class="card">

<h3>📊 About the Model</h3>

<p>
This application uses an LSTM Deep Learning model
to classify grocery demand into three categories:
</p>

<ul>
<li>🟢 Low Demand</li>
<li>🟡 Medium Demand</li>
<li>🔴 High Demand</li>
</ul>

<p>
The model uses historical sales information and
promotion/date-related features for demand classification.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# USER INPUT
# ============================================================

st.markdown("## 📝 Enter Prediction Information")


col1, col2 = st.columns(2)


with col1:

    sales = st.number_input(
        "Previous Sales",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    onpromotion = st.number_input(
        "Number of Items on Promotion",
        min_value=0,
        value=20,
        step=1
    )

    year = st.number_input(
        "Year",
        min_value=2013,
        max_value=2030,
        value=2017,
        step=1
    )


with col2:

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=8,
        step=1
    )

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=16,
        step=1
    )

    day_of_week = st.number_input(
        "Day of Week",
        min_value=0,
        max_value=6,
        value=2,
        step=1
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("")


if st.button(
    "🔮 Predict Demand",
    use_container_width=True
):

    try:

        # Create input array
        input_data = np.array([
            [
                sales,
                onpromotion,
                year,
                month,
                day,
                day_of_week
            ]
        ])


        # Scale input
        scaled_input = scaler.transform(
            input_data
        )


        # Create 30 time-step sequence
        sequence = np.repeat(
            scaled_input[np.newaxis, :, :],
            30,
            axis=1
        )


        # Prediction
        prediction = model.predict(
            sequence,
            verbose=0
        )


        predicted_class = np.argmax(
            prediction,
            axis=1
        )[0]


        confidence = (
            float(
                np.max(prediction)
            ) * 100
        )


        # Demand classes
        class_names = {
            0: "Low Demand",
            1: "Medium Demand",
            2: "High Demand"
        }


        demand = class_names[
            predicted_class
        ]


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            f"""
            <div class="result">
            🛒 Predicted Demand: {demand}
            </div>
            """,
            unsafe_allow_html=True
        )


        st.success(
            f"Prediction Confidence: {confidence:.2f}%"
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.markdown("### 📈 Demand Probabilities")


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "Low Demand",
                f"{prediction[0][0] * 100:.2f}%"
            )


        with c2:

            st.metric(
                "Medium Demand",
                f"{prediction[0][1] * 100:.2f}%"
            )


        with c3:

            st.metric(
                "High Demand",
                f"{prediction[0][2] * 100:.2f}%"
            )


        # ====================================================
        # BUSINESS INSIGHT
        # ====================================================

        st.markdown("### 💡 Business Insight")


        if predicted_class == 0:

            st.info(
                "Demand is predicted to be LOW. "
                "The store can avoid excessive stock "
                "and maintain essential inventory."
            )

        elif predicted_class == 1:

            st.warning(
                "Demand is predicted to be MEDIUM. "
                "The store should maintain a balanced "
                "inventory level."
            )

        else:

            st.error(
                "Demand is predicted to be HIGH. "
                "The store should prepare sufficient stock "
                "to reduce the risk of stock shortages."
            )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#000000;
        font-size:15px;
    ">
        <b>LSTM Deep Learning Model</b><br>
        Store 44 Grocery Demand Classification
    </div>
    """,
    unsafe_allow_html=True
)