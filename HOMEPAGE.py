# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 12:44:00 2023

@author: Divyanshu
"""

import streamlit as st
st.set_page_config(
    page_title='my project for samsung innovation campus',
    page_icon='🧠'
    )
st.title("main page")
st.sidebar.success("select above page")
import os

file_path = os.path.join("pages", "mycifar10.keras")

if os.path.exists(file_path):
    with open(file_path, "rb") as file:
        # Perform actions with the file
        pass
else:
    print("File not found at:", file_path)

