ARG FOLDER_FILES

# using a python small basic image
FROM python:3.7.10-slim

# creates a dir for our application /usr/local/certapp/uploads/OLEC.jpeg
WORKDIR /usr/local/certapp

# copy our requirements.txt file and install dependencies
COPY requirements.txt ./
COPY skills.json ./
COPY verticales.json ./
COPY names.txt ./
COPY places.txt ./

#UPDATE and RUN PIP
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

#Download Dictionary
RUN python -m spacy download en

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN python -m nltk.downloader words
RUN python -m nltk.downloader stopwords

RUN apt-get update -y
RUN apt-get -y install tesseract-ocr
RUN pip install opencv-contrib-python-headless
RUN pip install pillow
RUN pip install pytesseract

#CREATE FOLDER
ADD $FOLDER_FILES ./
# COPY app
COPY app.py ./

# run the application
CMD python -u app.py