import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from src.ai_journalist import AIJournalist
from src.db import DB
from datetime import datetime

from google.cloud import storage
import pandas as pd
import os


def main():
    st.title('🗞️ AI Journal')
    with st.form('my_form'):
        text = st.text_area('Enter here your short story:', )
        submitted = st.form_submit_button('Submit')

        openai_api_key = os.getenv('OPENAI_API_KEY')
        serper_api_key = os.getenv('SERPER_API_KEY')

        if submitted:
            # ai_journalist = AIJournalist(openai_api_key, serper_api_key)
            # response = ai_journalist.generate_response(text)
            
            response = 'response'

            st.info(response)
            now = datetime.now()
            # data to append db
            data_to_append = {
                            'datetime': f"{now}",
                            'user': f"{st.experimental_user.email}",
                            'input': f"{text}",
                            'output': f"{response}",
                            # 'log': ''
                             }

            # df = DB.read()
            # df_updated = DB.write(df, data_to_append)
            # DB.update()

            bucket_name = os.getenv('BUCKET_LOGS') 
            blob_name = os.getenv('FILE_LOGS') 
            client = storage.Client(project='gpt-news')
            bucket = client.get_bucket(bucket_name)

            blob = bucket.get_blob(blob_name)
            blob.download_to_filename(blob_name)
            df = pd.read_parquet(blob_name)

            new_entry_df = pd.DataFrame([data_to_append])
            df_new = pd.concat([df,new_entry_df])
            df_new.reset_index(inplace=True, drop=True)     
            df_new.to_parquet(blob_name)

            blob = bucket.get_blob(blob_name)         
            blob.upload_from_filename(blob_name)


         
if __name__ == "__main__":
    main()