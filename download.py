import requests
import os
import shutil
import subprocess

file_url = "https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip"
file = requests.get(file_url, stream=True)
with open("flower_data.zip", 'wb') as location:
	shutil.copyfileobj(file.raw, location)

process = subprocess.Popen(['unzip', 'flower_data.zip'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
output = process.stdout.read().decode("utf-8")