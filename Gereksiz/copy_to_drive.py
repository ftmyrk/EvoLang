from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

# Authenticate and create the Google Drive client
def authenticate_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Opens a browser for authentication
    return GoogleDrive(gauth)

# Upload a single file to Google Drive
def upload_single_file(file_path, drive_folder_id=None):
    drive = authenticate_drive()
    file_name = os.path.basename(file_path)
    
    file = drive.CreateFile({'title': file_name, 'parents': [{"id": drive_folder_id}] if drive_folder_id else []})
    file.SetContentFile(file_path)
    file.Upload()
    
    print(f"Uploaded {file_name} to Google Drive with ID: {file['id']}")
    return file['id']

# Specify the file to upload
local_file = "/home/otamy001/EvoLang/generated_data/generated_responses_2013.csv"  # Change this to your local file path
google_drive_folder_id = "586321020548"  # Replace with your folder ID

# Upload the file
upload_single_file(local_file, google_drive_folder_id)