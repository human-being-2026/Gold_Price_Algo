import streamlit as st
import os
import papermill as pm

st.title("My First App")
user_input = st.text_input("Enter some text")
st.write("You entered:", user_input)
