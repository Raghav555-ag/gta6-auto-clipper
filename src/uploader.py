"""YouTube Data API Shorts uploader stub."""
import os
from pathlib import Path


class YouTubeUploader:
    def __init__(self, client_secrets_file: str = "client_secrets.json"):
        self.client_secrets_file = client_secrets_file

    def upload_short(self, video_path: str, title: str, description: str, tags: list = None) -> str:
        """Upload a video as a YouTube Short."""
        if not os.path.exists(self.client_secrets_file):
            print(f"[Warning] {self.client_secrets_file} not found. Running in simulation mode.")
            print(f"[Simulated Upload] Title: {title} | File: {video_path}")
            return "simulated_video_id"
        
        # Real Google API upload flow
        # In actual usage: googleapiclient.discovery.build('youtube', 'v3', credentials=creds)
        print(f"Uploading {video_path} to YouTube Shorts...")
        return "uploaded_video_id"
