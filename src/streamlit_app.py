import os
from dotenv import load_dotenv
load_dotenv()
openai_key = os.getenv('openai_key')
serper_api_key = os.getenv('serper_api_key')
import streamlit as st
from src.ai_journalist import AIJournalist

def main():
    st.title('🗞️ AI Journal')
    with st.form('my_form'):
        text = st.text_area('Enter here your short story:', )
        submitted = st.form_submit_button('Submit')

        openai_key = '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'  # news
        serper_api_key = 'a395f76722402c3a8b434379cd3741ba8ea13c42'

        if submitted:
            ai_journalist = AIJournalist(openai_key, serper_api_key)
            response = ai_journalist.generate_response(text)
            st.info(response)


if __name__ == "__main__":
    main()