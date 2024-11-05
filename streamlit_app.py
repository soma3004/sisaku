import streamlit as st
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

def init_db():
    conn = sqlite3.connect('user.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT)''')
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect('user.db')
    c = conn.cursor()
    hashed_password = generate_password_hash(password)
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
    conn.commit()
    conn.close()

def check_user(username, password):
    conn = sqlite3.connect('user.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return check_password_hash(result[0], password)
    return False

def login():
    st.title('ログイン')

    username = st.text_input('ユーザー名')
    password = st.text_input('パスワード', type='password')

    if st.button('ログイン'):
        if check_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success('ログイン成功！')
        else:
            st.error('ユーザー名またはパスワードが間違っています。')

def logout():
    if 'logged_in' in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.success('ログアウトしました。')

def main():
    # セッション状態の初期化
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        st.write(f"ようこそ, {st.session_state.username}さん！")
        if st.button('ログアウト'):
            logout()
    else:
        login()
if __name__ == '__main__':
    init_db()  # データベースの初期化（最初に1回だけ実行すればOK）
    main()
