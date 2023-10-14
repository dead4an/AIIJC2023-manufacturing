# Сервер и система
import streamlit as st
from streamlit.web.cli import main
import os
import gc
import tracemalloc
tracemalloc.start()

# Текст
from helpers.texts import HOME_PAGE_CONTENT

# Пути
ROOT = os.getcwd()
TRAIN_DATASET = os.path.join(ROOT, 'data/train_AIC.csv')
BALANCED_DATASET = os.path.join(ROOT, 'data/balanced_train.csv')
TEST_DATASET = os.path.join(ROOT, 'data/test_AIC.csv')
SUBMISSION_PATH = os.path.join(ROOT, 'submissions/')
MODEL_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_model.dat')
PREC_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_preprocessor.dat')

st.set_page_config('Home')

def main():
    # Header
    _, col1, col2, _ = st.columns([0.4, 0.3, 0.4, 0.7], gap='small')
    with col1:
        st.image('./streamlit/logo.jpg', width=100)

    with col2:
        st.markdown("<h1 style='text-align: left; color: black;'>UnThinkable</h1>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: center; color: black;'>Аналитическая платформа by Team UnThinkable</h4>", unsafe_allow_html=True)
    # st.text('Аналитическая платформа by Team UnThinkable')
    st.divider()

    # Content
    st.text(HOME_PAGE_CONTENT)
    
if __name__ == '__main__':
    main()
    gc.collect()

usage = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(usage[0] >> 20, usage[1] >> 20)
