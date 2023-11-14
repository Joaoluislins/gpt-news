import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from src.ai_journalist import AIJournalist

def main():
    st.title('🗞️ AI Journal')
    with st.form('my_form'):
        text = st.text_area('Enter here your short story:', )
        submitted = st.form_submit_button('Submit')

        openai_api_key = os.getenv('OPENAI_API_KEY')
        serper_api_key = os.getenv('SERPER_API_KEY')

        if submitted:
            ai_journalist = AIJournalist(openai_api_key, serper_api_key)
            response = ai_journalist.generate_response(text)
            st.info(response)


if __name__ == "__main__":
    main()