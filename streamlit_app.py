import streamlit as st
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# データベースの初期化
def init_db():
    try:
        conn = sqlite3.connect('user.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE,
                        password TEXT)''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        st.error(f"データベースの初期化に失敗しました: {e}")

# ユーザーの追加
def add_user(username, password):
    try:
        conn = sqlite3.connect('user.db')
        c = conn.cursor()
        hashed_password = generate_password_hash(password)
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        conn.close()
        st.success('ユーザー登録が完了しました！')
    except sqlite3.IntegrityError:
        st.error('このユーザー名はすでに使用されています。')
    except sqlite3.Error as e:
        st.error(f"ユーザー登録に失敗しました: {e}")

# ユーザーの認証
def check_user(username, password):
    try:
        conn = sqlite3.connect('user.db')
        c = conn.cursor()
        c.execute('SELECT password FROM users WHERE username = ?', (username,))
        result = c.fetchone()
        conn.close()
        if result:
            return check_password_hash(result[0], password)
        return False
    except sqlite3.Error as e:
        st.error(f"認証エラー: {e}")
        return False

# ログイン処理
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

# ログアウト処理
def logout():
    if 'logged_in' in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.success('ログアウトしました。')

# 新規ユーザー登録処理
def register():
    st.title('新規ユーザー登録')

    username = st.text_input('ユーザー名')
    password = st.text_input('パスワード', type='password')
    confirm_password = st.text_input('パスワード（確認）', type='password')

    if password != confirm_password:
        st.error('パスワードが一致しません。')

    if st.button('登録'):
        if username and password:
            add_user(username, password)
        else:
            st.error('ユーザー名とパスワードを入力してください。')

# メイン関数
def main():
    # セッション状態の初期化
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        st.write(f"ようこそ, {st.session_state.username}さん！")
        if st.button('ログアウト'):
            logout()
    else:
        page = st.radio('ページを選択', ('ログイン', '新規登録'))
        if page == 'ログイン':
            login()
        else:
            register()

if __name__ == '__main__':
    init_db()  # データベースの初期化（最初に1回だけ実行すればOK）
    main()
