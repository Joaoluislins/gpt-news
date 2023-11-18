from google.cloud import storage
import pandas as pd
import os



class DB:
    def __init__(self):
        self.bucket_name = os.getenv('BUCKET_LOGS') 
        self.blob_name = os.getenv('FILE_LOGS') 
        self.client = storage.Client(project='gpt-news')
        self.bucket = self.client.get_bucket(self.bucket_name)
        

    def read(self):
        blob = self.bucket.get_blob(self.blob_name)
        blob.download_to_filename(self.blob_name)
        df = pd.read_parquet(self.blob_name)
        return df

        
    def write(self, df, data_to_append):
        new_entry_df = pd.DataFrame([data_to_append])
        df_new = pd.concat([df,new_entry_df])
        df_new.reset_index(inplace=True, drop=True)     
        df_new.to_parquet(self.blob_name)

        return df_new
        
    def update(self):
        blob = self.bucket.get_blob(self.blob_name)         
        blob.upload_from_filename(self.blob_name)