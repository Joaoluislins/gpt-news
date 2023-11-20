import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from src.ai_journalist import AIJournalist
from src.db import DB
from datetime import datetime
import pandas as pd
import os
from datetime import datetime
import pytz


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

            response = 'response'

            st.info(response)
            
            now = datetime.now(tz=pytz.timezone('US/Eastern'))
            # data to append db
            data_to_append = {
                            'datetime': f"{now}",
                            'user': f"{st.experimental_user.email}",
                            'input': f"{text}",
                            'output': f"{response}",
                            # 'log': ''
                             }

            db = DB()
            df = db.read()
            df_updated = db.write(df, data_to_append)
            db.update()
          
         
if __name__ == "__main__":
    main()