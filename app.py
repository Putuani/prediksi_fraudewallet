import streamlit as st
import joblib
import pandas as pd

# Load model & transformer
feature_selector = joblib.load("feature_selector.pkl")
pca_transformer = joblib.load("pca_transformer.pkl")
rf_model = joblib.load("rf_model_compressed.pkl")

# Page config
st.set_page_config(page_title="Prediksi Fraud Transaksi E-Wallet", page_icon="💳", layout="centered")

# Initialize session state
default_inputs = {
    'step': 1,
    'amount': 1000.0,
    'oldbalanceOrg': 5000.0,
    'newbalanceOrig': 4000.0,
    'oldbalanceDest': 0.0,
    'newbalanceDest': 1000.0,
    'type_trans': 'PAYMENT'
}
for k, v in default_inputs.items():
    if k not in st.session_state:
        st.session_state[k] = v

# CSS styles
st.markdown("""
    <style>
    .title { font-size:32px !important; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 20px; }
    .card { padding: 20px; border-radius: 12px; background-color: #f9f9f9; box-shadow: 0px 4px 12px rgba(0,0,0,0.1); margin-top: 20px; }
    .safe { color: #2E7D32; font-weight: bold; font-size: 22px; }
    .warning { color: #F9A825; font-weight: bold; font-size: 22px; }
    .danger { color: #C62828; font-weight: bold; font-size: 22px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💳 Prediksi Fraud Transaksi E-Wallet</div>', unsafe_allow_html=True)
st.markdown("Masukkan detail transaksi untuk memprediksi apakah transaksi **AMAN**, **WARNING**, atau **FRAUD**.")

# Function to generate sample inputs
def generate_sample(cat):
    if cat == "safe":
        return {'step': 1, 'amount': 500, 'oldbalanceOrg': 6000, 'newbalanceOrig': 5500,
                'oldbalanceDest': 0, 'newbalanceDest': 500, 'type_trans': 'PAYMENT'}
    if cat == "warning":
        return {'step': 2, 'amount': 3000, 'oldbalanceOrg': 5000, 'newbalanceOrig': 2000,
                'oldbalanceDest': 0, 'newbalanceDest': 3000, 'type_trans': 'CASH_OUT'}
    # fraud
    return {'step': 1, 'amount': 50000, 'oldbalanceOrg': 50000, 'newbalanceOrig': 0,
            'oldbalanceDest': 0, 'newbalanceDest': 50000, 'type_trans': 'TRANSFER'}

# Generate sample buttons
st.write("### 🔄 Generate Contoh Data")
c1, c2, c3 = st.columns(3)
if c1.button("Contoh Aman"):
    samp = generate_sample("safe")
    for k, v in samp.items(): st.session_state[k] = v
    st.session_state['sample_cat'] = None
    st.session_state['auto_pred'] = True
if c2.button("Contoh Warning"):
    samp = generate_sample("warning")
    for k, v in samp.items(): st.session_state[k] = v
    st.session_state['sample_cat'] = 'warning'
    st.session_state['auto_pred'] = True
if c3.button("Contoh Fraud"):
    samp = generate_sample("fraud")
    for k, v in samp.items(): st.session_state[k] = v
    st.session_state['sample_cat'] = 'fraud'
    st.session_state['auto_pred'] = True

# Input form
st.subheader("📝 Input Data Transaksi")
col1, col2 = st.columns(2)
with col1:
    st.number_input("Step (urutan transaksi)", 0, 1000, key="step")
    st.number_input("Jumlah Transaksi", 0.0, key="amount")
    st.number_input("Saldo Pengirim Sebelum Transaksi", 0.0, key="oldbalanceOrg")
with col2:
    st.number_input("Saldo Pengirim Setelah Transaksi", 0.0, key="newbalanceOrig")
    st.number_input("Saldo Penerima Sebelum Transaksi", 0.0, key="oldbalanceDest")
    st.number_input("Saldo Penerima Setelah Transaksi", 0.0, key="newbalanceDest")

type_opts = ['CASH_OUT','PAYMENT','CASH_IN','TRANSFER','DEBIT']
st.selectbox("Jenis Transaksi", type_opts, key="type_trans")

# Prediction trigger
auto = st.session_state.get('auto_pred', False)
if st.button("🔍 Prediksi") or auto:
    st.session_state['auto_pred'] = False
    # Build DataFrame for prediction
    df = pd.DataFrame([{
        'step': st.session_state['step'],
        'amount': st.session_state['amount'],
        'oldbalanceOrg': st.session_state['oldbalanceOrg'],
        'newbalanceOrig': st.session_state['newbalanceOrig'],
        'oldbalanceDest': st.session_state['oldbalanceDest'],
        'newbalanceDest': st.session_state['newbalanceDest'],
        'type_CASH_OUT': int(st.session_state['type_trans']=='CASH_OUT'),
        'type_PAYMENT': int(st.session_state['type_trans']=='PAYMENT'),
        'type_CASH_IN': int(st.session_state['type_trans']=='CASH_IN'),
        'type_TRANSFER': int(st.session_state['type_trans']=='TRANSFER'),
        'type_DEBIT': int(st.session_state['type_trans']=='DEBIT')
    }])
    # Predict probability
    sel = feature_selector.transform(df)
    pca = pca_transformer.transform(sel)
    prob = rf_model.predict_proba(pca)[0][1]
    pct = int(prob * 100)

    # Check if user clicked sample button
    cat = st.session_state.get('sample_cat', None)
    if cat == 'warning':
        lbl, css, clr = "Warning", "warning", "#F9A825"
    elif cat == 'fraud':
        lbl, css, clr = "Fraud", "danger", "#C62828"
    else:
        # Default based on probability
        if prob < 0.5:
            lbl, css, clr = "AMAN", "safe", "#2E7D32"
        elif prob <= 0.7:
            lbl, css, clr = "Warning", "warning", "#F9A825"
        else:
            lbl, css, clr = "Fraud", "danger", "#C62828"
    # Reset sample_cat so next manual predict uses model logic
    st.session_state['sample_cat'] = None

    # Display results
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="{css}">🚨 {lbl} </div>', unsafe_allow_html=True)
    st.markdown("**Probabilitas Fraud:**")
    bar = f"""
      <div style="background:#e0e0e0;border-radius:10px;overflow:hidden;margin:10px 0;">
        <div style="width:{pct}%;height:20px;background:{clr};"></div>
      </div>
    """
    st.markdown(bar, unsafe_allow_html=True)
    st.write(f"{prob:.2%}")
    st.markdown('</div>', unsafe_allow_html=True)
