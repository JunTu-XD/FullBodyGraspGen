#!/bin/bash
# cd scratch pla

# Download zip dataset from Google Drive
filename='save_pics.zip'  #替换
fileid='1zAHCV6_fLEb4iuDpuwomSLrbmt5pe9bJ'  #替换
	wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt