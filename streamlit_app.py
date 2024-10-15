import streamlit as st
import pandas as pd
import sqlite3
import hashlib
from PIL import Image

# データベース接続
def get_db_connection():
    conn = sqlite3.connect('database.db')
    return conn

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def create_user_table():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT PRIMARY KEY, password TEXT, icon BLOB)')
    conn.commit()
    conn.close()

def add_user(username, password, icon):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO userstable(username, password, icon) VALUES (?, ?, ?)', (username, password, icon))
        conn.commit()
    except sqlite3.IntegrityError:
        return False  # ユーザー名が既に存在する
    finally:
        conn.close()
    return True

def login_user(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username = ? AND password = ?', (username, password))
    data = c.fetchall()
    conn.close()
    return data

def get_user_icon(username):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT icon FROM userstable WHERE username = ?', (username,))
    icon = c.fetchone()
    conn.close()
    return icon[0] if icon else None

def main():
    st.title("ログイン機能テスト")

    menu = ["ホーム", "ログイン", "サインアップ", "ユーザー設定"]
    choice = st.sidebar.selectbox("メニュー", menu)

    create_user_table()

    if choice == "ホーム":
        st.subheader("ホーム画面です")

    elif choice == "ログイン":
        st.subheader("ログイン画面です")

        username = st.sidebar.text_input("ユーザー名を入力してください")
        password = st.sidebar.text_input("パスワードを入力してください", type='password')
        if st.sidebar.checkbox("ログイン"):
            hashed_pswd = make_hashes(password)

            result = login_user(username, hashed_pswd)
            if result:
                st.success("{}さんでログインしました".format(username))
                icon = get_user_icon(username)
                if icon:
                    st.image(icon, width=100)  # アイコンを表示
            else:
                st.warning("ユーザー名かパスワードが間違っています")

    elif choice == "サインアップ":
        st.subheader("新しいアカウントを作成します")
        new_user = st.text_input("ユーザー名を入力してください")
        new_password = st.text_input("パスワードを入力してください", type='password')
        icon_file = st.file_uploader("アイコンをアップロードしてください", type=['png', 'jpg', 'jpeg'])

        if st.button("サインアップ"):
            if icon_file is not None:
                icon = icon_file.read()  # アイコンをバイナリ形式で読み込む
                if add_user(new_user, make_hashes(new_password), icon):
                    st.success("アカウントの作成に成功しました")
                    st.info("ログイン画面からログインしてください")
                else:
                    st.warning("このユーザー名はすでに存在します")
            else:
                st.warning("アイコンをアップロードしてください")

    elif choice == "ユーザー設定":
        st.subheader("ユーザー設定")
        username = st.sidebar.text_input("ユーザー名を入力してください")
        new_password = st.sidebar.text_input("新しいパスワードを入力してください", type='password')
        new_icon_file = st.file_uploader("新しいアイコンをアップロードしてください", type=['png', 'jpg', 'jpeg'])

        if st.button("更新"):
            if new_icon_file is not None:
                new_icon = new_icon_file.read()
                conn = get_db_connection()
                c = conn.cursor()
                c.execute('UPDATE userstable SET password = ?, icon = ? WHERE username = ?', (make_hashes(new_password), new_icon, username))
                conn.commit()
                conn.close()
                st.success("ユーザー設定が更新されました")
            else:
                st.warning("新しいアイコンをアップロードしてください")

if __name__ == '__main__':
    main()
