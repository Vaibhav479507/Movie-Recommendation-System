import streamlit as st
import joblib
import re
import requests
import base64


# Background Image Function

def get_base64(img_path):
    with open(img_path, "rb") as img:
        return base64.b64encode(img.read()).decode()


def set_background():

    img_path = r"C:\Users\AsusT\OneDrive\Desktop\ML\movie_poster.png"

    img_base64 = get_base64(img_path)

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image:
                linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
                url("data:image/png;base64,{img_base64}");

            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        h1, h2, h3, h4, h5, h6, p, label {{
            color: white !important;
        }}

        .stSelectbox label {{
            color: white !important;
            font-size: 20px;
            font-weight: bold;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


set_background()


# Text Cleaning Function

def mycleaning(doc):
    return re.sub("[^a-zA-Z0-9 ]", "", doc).lower()


# Load Files

df = joblib.load('dataset.pkl')
model = joblib.load('movie_model.pkl')
tv = joblib.load('movie_vectorizer.pkl')


# Title

st.title("🎬 Movie Recommendation System")


# Movie Dropdown

movie = st.selectbox(
    'Select a Movie',
    df.name,
    key="movie_select"
)


# Check Button

check = st.button(
    "Check",
    key="check_button"
)



# Recommendation Logic


if check:

    index = df[df.name == movie].index[0]

    vector = tv.transform([df.loc[index].values[2]])

    distances, indexes = model.kneighbors(
        vector,
        n_neighbors=6
    )

    mid = df.loc[indexes[0][1:]].movie_id.values
    mname = df.loc[indexes[0][1:]].name.values

    st.subheader("Recommended Movies")

    for m, i in zip(mname, mid):

        st.write(f"### {m}")

        api = f"http://www.omdbapi.com/?i={i}&apikey=471e6e4b"

        resp = requests.get(api)

        data = resp.json()

        poster = data.get('Poster')

        if poster and poster != "N/A":

            st.image(
                poster,
                width=250
            )

        else:
            st.write("Poster not available")
