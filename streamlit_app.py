import streamlit as st

# アプリのタイトル
st.title("ユーザー設定アプリ")

# ユーザーからの入力を受け取る
with st.form("settings_form"):
    user_name = st.text_input("お名前を入力してください")
    user_color = st.color_picker("好きな色を選んでください", "#00f900")
    submit_button = st.form_submit_button("設定を保存")

# 設定が保存された場合の処理
if submit_button:
    st.success(f"設定が保存されました！お名前: {user_name}, 好きな色: {user_color}")
    st.markdown(f"<h1 style='color: {user_color};'>こんにちは、{user_name}さん！</h1>", unsafe_allow_html=True)
