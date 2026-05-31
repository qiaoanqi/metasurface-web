import streamlit as st
st.set_page_config(page_title="Test", layout="wide")
st.title("Hello World")
st.write("If you see this, Streamlit Cloud works!")
st.write(f"Python: {__import__('sys').version}")
st.write(f"Streamlit: {st.__version__}")
try:
    import numpy as np; st.write(f"NumPy: {np.__version__}")
except: st.write("NumPy: FAILED")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; st.write(f"Matplotlib: {matplotlib.__version__}")
except Exception as e: st.write(f"Matplotlib: FAILED - {e}")
try:
    from PIL import Image; st.write(f"Pillow: OK")
except: st.write("Pillow: FAILED")
