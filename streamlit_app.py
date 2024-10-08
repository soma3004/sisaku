import streamlit as st
import pandas as pd
import sqlite3
import hashlid

conn = sqlite3.connect('database.db')
c = conn.cursor()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False    