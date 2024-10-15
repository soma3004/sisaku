import streamlit as st

# ユーザー名とパスワードの設定
USER_CREDENTIALS = {
    "user1": "password123",
    "user2": "password456"
}

# ログインセッションを管理するためのセッション状態
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# ログインフォーム
if not st.session_state.logged_in:
    st.title("ログイン")

    username = st.text_input("ユーザー名")
    password = st.text_input("パスワード", type='password')

    if st.button("ログイン"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("ログイン成功！")
        else:
            st.error("ユーザー名またはパスワードが正しくありません。")
else:
    # ログイン後の画面
    st.title(f"ようこそ、{st.session_state.username}さん！")

    # アイコンの表示
    icon_url = "https://example.com/path/to/your/icon.png"  # アイコンのURLを指定
    st.image(icon_url, width=100)  # アイコンを表示
    st.write("ここにアプリのコンテンツを追加します。")

    if st.button("ログアウト"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("ログアウトしました。")
