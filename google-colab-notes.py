# Upload File into Colab

from google.colab import files
uploaded = files.upload()

#-----------------------------------------------
# Download file to local

files.download('Week4.ipynb')

#-----------------------------------------------
# Mount Google Drive as a Folder

from google.colab import drive
drive.mount('/my/gdrive',force_remount=True)
