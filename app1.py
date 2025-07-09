import streamlit as st

st.title("✅ Streamlit App is Running!")

st.write("This is a test app. If you see this, your deployment worked!")

# Optional: Add an input
name = st.text_input("What's your name?")
if name:
    st.success(f"Hello, {name}!")
