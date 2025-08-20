import streamlit_authenticator as stauth

passwords_to_hash = ["password123", "password456"]
hashed_passwords = stauth.Hasher(passwords_to_hash).generate()
print(hashed_passwords)
