from flask import Flask, request, redirect, render_template,jsonify
from pytesseract.pytesseract import file_to_dict
from flask_cors import CORS
import logging, json, time
import shutil
import os
import urllib.request
from werkzeug.utils import secure_filename
import docx2txt
import spacy
import re
import fitz
import pytesseract
import cv2
import tempfile
import dateparser
import datefinder
import pandas as pd
from PIL import Image
from unicodedata import normalize
from dateparser.search import search_dates
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter
from gensim.summarization import keywords
from spacy.matcher import PhraseMatcher

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
stopwords.words('spanish')
stopwords.words('english')

#python -m nltk.downloader stopwords

en_nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(en_nlp.vocab)

PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')

def load_json_file(_file):
    with open(_file, encoding="utf8") as f:
        data = json.load(f)
    return data

SKILLS_DB = load_json_file('./skills.json')
VERTICALES_DB = load_json_file('./verticales.json')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def count_vectorizer_text(text_list):
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text_list)
    matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
    return round(matchPercentage,2)

def _400(data):
    resp = jsonify(data)
    resp.status_code = 400
    return resp

def _201(data):
    resp = jsonify(data)
    resp.status_code = 201
    return resp

def _500(data):
    resp = jsonify(data)
    resp.status_code = 500
    return resp


def extractPDFFiles(_file):
    with fitz.open(_file) as doc:
        text = ""
        for page in doc:
            text += page.getText().strip()
        text = text.replace("\n"," ")
        return text

def extractPDFFiles_back(_file):
    pdfFileObj = open(_file, 'rb')
    pdfReader = PyPDF4.PdfFileReader(pdfFileObj)
    n_pages = pdfReader.getNumPages()
    this_doc = ''
    for i in range(n_pages):
        pageObj = pdfReader.getPage(i)
        this_text = pageObj.extractText()
        this_doc += this_text
    return this_doc

def clean_text(corpus):
    cleaned_text = ""
    for i in corpus:
        cleaned_text = cleaned_text + i.lower().replace("'", "").replace('\t', ' ').replace('\n', ' ')
    return cleaned_text

def get_name_from_file(file):
    base = os.path.basename(file)
    return os.path.splitext(base)

def call_process_by_file(_file):
    name, ext = get_name_from_file(_file)
    if ext == '.pdf':
        try:
            return extractPDFFiles(_file)
        except:
            print("An exception occurred, PLEASE review with Administrator, filename:{}, name: {}".format(_file, name))
    elif ext == '.docx':
        try:
            return clean_text(docx2txt.process(_file))
        except:
            print("An exception occurred, PLEASE review with Administrator, filename:{}, name: {}".format(_file, name))
    elif ext == '.jpeg' or ext == '.png' or ext == '.JPG':
        try:
            return extract_text_from_image(_file)
        except:
            print("An exception occurred, PLEASE review with Administrator, filename:{}, name: {}".format(_file, name))
    elif ext == 'doc':
        return "No Soportado"
    
def extract_terms_by_job(_job):
    terms = keywords(_job, ratio=0.25).split('\n')
    return terms
    
def match_terms_by_resumen_and_job(_text_resume, _job_terms):
    #only run npl.make_doc to spend thinhs up
    patterns = [en_nlp.make_doc(t) for t in _job_terms]
    matcher.add("Spec", patterns)
    doc = en_nlp(_text_resume)
    matchkeywords = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        if len(span.text) > 3:
            matchkeywords.append(span.text)
    return Counter(matchkeywords)

def find_skill(_skill):
    skill_found=False
    if not (SKILLS_DB.get(_skill) is None):
        skill_found = True
    return skill_found

def in_substring(given, sub):
    return sub in given

def extract_names(resume_text):
    person_names = []
    for sent in nltk.sent_tokenize(resume_text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                person_names.append(
                    ' '.join(chunk_leave[0] for chunk_leave in chunk.leaves())
                )
    return person_names

def extract_phone_number(resume_text):
    phone = re.findall(PHONE_REG, resume_text)
    if phone:
        number = ''.join(phone[0])
        if resume_text.find(number) >= 0 and len(number) < 16:
            return number
    return None

def extract_emails(resume_text):
    return re.findall(EMAIL_REG, resume_text)
    
def extract_skills(input_text):
    word_tokens = nltk.tokenize.word_tokenize(input_text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]
    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]
    # generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
    # we create a set to keep the results in.
    found_skills = []
    # we search for each token in our skills database
    for token in filtered_tokens:
        if find_skill(token.lower()):
            found_skills.append(token)
    # we search for each bigram and trigram in our skills database
    for ngram in bigrams_trigrams:
        if find_skill(ngram.lower()):
            found_skills.append(ngram)
    return found_skills  

def get_human_names(text):
    person_list = []
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary = False)
    #print("DATA ATTRS {}".format(sentt))
    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        #print("DATA PERSON {}".format(person))
        if len(person) > 1: #avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
        person = []
    return person_list

def resume_details(resume_text):
    details=[]
    # Get Names
    name = extract_names(resume_text)
    if len(name) < 0:
        name=[]
    #details.append(name)
    # Get phone
    phone = extract_phone_number(resume_text)
    if not phone:
        phone ='0'
    #details.append(phone)
    # Get email ( first one ingore others email[0])
    email = extract_emails(resume_text)

    #details.append(email)
    # Get skills
    skills = extract_skills(resume_text)
    if not skills:
        skills=[]
    #details.append(skills)
    
    industria = extract_vertical(resume_text)
    if not industria:
        industria=[]
    
    details.append({"name":name,"phone":phone,"email":email, "skills": skills, "industria":industria})
    return details

def extract_vertical(input_text):
    organizations = []

    # first get all the organization names using nltk
    for sent in nltk.sent_tokenize(input_text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'ORGANIZATION':
                organizations.append(' '.join(c[0] for c in chunk.leaves()))
    #set dictionary
    industria = []
    for org in organizations:
        print("organizaciones {} ".format(org))
        for key in VERTICALES_DB:
            for client in VERTICALES_DB[key].get('client'):
                if in_substring(client.lower(), org.lower()):
                    industria.append(key) 
    return list(set(industria))
  
def extract_text_from_image(file):
    img = cv2.imread(file)
    # Remove background
    # Grey Scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binarizacion
    bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # Color
    col = cv2.cvtColor(bin,cv2.COLOR_BGR2RGB)
    # Denoising
    denoise = cv2.fastNlMeansDenoisingColored(col, None, 10, 10, 7, 21)
    #logging.warning("Image2: "+ denoise)
    # Extract text
    custom_config = r'--oem 3 --psm 6'
    text = str(pytesseract.image_to_string(denoise, config=custom_config))
    # Clean text
    # Spaces
    textwr = ""
    for lane in text.split('\n'):
        length_word = 0
        for word in lane.split():
            length_word += len(text)
        if (length_word >= 5):
            textwr = textwr+lane+" "
    return textwr

def extract_date(text):
    formats = ['%d de %B de %Y','%d de %B del %Y','%d-%B-%Y','%d/%B/%Y','%d - %B - %Y','%d / %B / %Y', # %B Month as locale’s full name, spanish format.
           '%d de %b de %Y','%d de %b del %Y','%d-%b-%Y','%d/%b/%Y','%d - %b - %Y','%d / %b / %Y', # %b Month as locale’s abbreviated name, spanish format.
           '%d-%m-%Y','%d/%m/%Y','%d - %m - %Y','%d / %m / %Y', # %m Month as a zero-padded decimal number, spanish format.
           '%B %d del %Y','%B %d-%Y','%B %d/%Y', '%B %d - %Y','%B %d / %Y', # Differents spanish formats.
           '%m-%d-%Y','%m/%d/%Y','%m - %d - %Y','%m / %d / %Y',  # %m Month as a zero-padded decimal number, english format.
           '%B %d-%Y','%B %d/%Y','%B %d - %Y','%B %d / %Y', # %B Month as locale’s full name, english format.
           '%b %d-%Y','%b %d/%Y','%b %d - %Y','%b %d / %Y'] # %b Month as locale’s abbreviated name, english format.
    language = ['es','en']
    dates = search_dates(text, languages=language, settings={'STRICT_PARSING': True})
    if dates == None:
        return extract_date1(text, language, formats)
    else:
        for day in dates:
            date = day[0]
        #dateesp = dateparser.parse(date, date_formats=formats, languages=language).strftime("%d/%m/%Y")
        dateen = dateparser.parse(date, date_formats=formats, languages=language).strftime("%m/%d/%Y")
        date = "EN: " + dateen
    return date

def extract_date1(txt, language, formats):
    dt = datefinder.DateFinder()
    matches = list(dt.extract_date_strings(txt, strict=True))
    date = None
    for match in matches:
        date = match[0]
    if date == None:
        return extract_date2(txt, language, formats)
    else:
        dates = search_dates(txt, languages=language, settings={'STRICT_PARSING': True})
        if dates == None:
            return extract_date2(txt, language, formats)
        else:
            for day in dates:
                date = day[0]
            #dateesp = dateparser.parse(sdate, date_formats=formats, languages=language).strftime("%d/%m/%Y")
            dateen = dateparser.parse(date, date_formats=formats, languages=language).strftime("%m/%d/%Y")
            date = "EN: " + dateen
        return date

def extract_date2(txt, language, formats):
    dt = datefinder.DateFinder()
    matches = list(dt.find_dates(txt))
    date = None
    for match in matches:
        date = match
    if date == None:
        date="No Dates Found"
        return date
    else:
        date = str(date)
        chdate = " 00:00:00"
        date = date.replace(chdate,"")
        dates = search_dates(date, languages=language, settings={'STRICT_PARSING': True})
        if dates == None:
            date="No Dates Found"
            return date
        else:
            for day in dates:
                date = day[0]
            #dateesp = dateparser.parse(date, date_formats=formats, languages=language).strftime("%d/%m/%Y")
            dateen = dateparser.parse(date, date_formats=formats, languages=language).strftime("%m/%d/%Y")
            date = "EN: " + dateen
            return date

def extract_place(text):
    filep = open('./places.txt', 'r', encoding='UTF-8')
    locations = filep.readlines()
    f = lambda x: x.replace('\n', '')
    locations = list(map(f, locations))

    newtext = text.split(' ')
    places = []
    for text in newtext:
        for location in locations:
            if (text.startswith(location) == True) and (text.endswith(location) == True):
                places.append(location)
    
    if (len(places) == 0):
        result = "No place founded"
    if (len(places) == 1):
        result = places[0]
    if (len(places) > 1):      
        sw = True
        for place in places:
            if sw == False:
                result = result + ", " + place
            if sw == True:
                result = place
                sw = False
    return result

def extract_person_full_name(text):
    textwrr = ''.join([i for i in text if not i.isdigit()])
    tokens = re.split('\s+', textwrr)
    stopwords = nltk.corpus.stopwords.words('spanish')
    stopworde = nltk.corpus.stopwords.words('english')
    words = [word for word in tokens if word not in stopwords and word not in stopworde]

    file = open('./names.txt', 'r', encoding='UTF-8')
    names = file.readlines()
    f = lambda x: x.replace('\n', '')
    names = list(map(f, names))

    name_list = []
    for word in words:
        for name in names:
            if name == word:
                name_list.append(name)
    temp = ""
    for word in words:
        temp = temp + word + " "
    person_names = []
    tokens = nltk.tokenize.word_tokenize(temp)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary = False)
    #print("DATA ATTRS {}".format(sentt))
    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        #print("DATA PERSON {}".format(person))
        if len(person) > 1: #avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_names:
                person_names.append(name[:-1])
            name = ''
        person = []
    person_namess = []
    for sent in nltk.sent_tokenize(temp):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                person_namess.append(' '.join(chunk_leave[0] for chunk_leave in chunk.leaves()))
    for person in person_namess:
        if person not in person_names:
            person_names.append(person)
    full_name = []
    for person in person_names:
        for namel in name_list:
            if ((namel in person) == True):
                full_name.append(person)
    matches = []
    for match in full_name:
        matches.append(full_name.count(match))
    if not matches:
        fullname = "No names found"
        return fullname
    may = 0
    for match in matches:
        if may < match:
            may = match
    fullname = full_name[matches.index(may)]
    return fullname

def extract_institute_organization_certification(text):
    organizations = []
    # first get all the organization names using nltk
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'ORGANIZATION':
                organizations.append(' '.join(c[0] for c in chunk.leaves()))
    #set dictionary
    industria = []
    for org in organizations:
        for key in VERTICALES_DB:
            for client in VERTICALES_DB[key].get('client'):
                if in_substring(client.lower(), org.lower()):
                    industria.append(key)
    sw = True
    cert = "No certifications found"
    for inst in organizations:
        if sw == False:
            cert = cert + ", " + inst   
        if sw == True:
            cert = inst
            sw = False
    return cert

app = Flask(__name__)
CORS(app)

app.secret_key = 'A1cnUlI7PItPrUKp5eIvPrDAkvoCDOYk'#secrets.token_urlsafe(16)
#It will allow below 16MB contents only
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()

# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['pdf', 'docx','jpeg','jpg','png'])

@app.route('/upload', methods=['POST'])
def upload_file():
    job = request.form.get('job')
    if request.method == 'POST':
        if 'files[]' not in request.files:
            return _400({'message' : 'No file part in the request'})

        if not job or not job.strip():
            return _400({'message' : 'No Data Job Description in the request'})
        files = request.files.getlist('files[]')
        errors = {}
        success = False
        process_data=[]
        
        for file in files:
            if file and allowed_file(file.filename):                                
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(path)
                except:
                    logging.error("Error al salvar el archivo")
                text_resume = call_process_by_file(path)
                if text_resume == None:
                    errors['message'] = 'File {} : cant extract data'.format(file.filename)
                    return errors
                else:
                    date = extract_date(text_resume)
                    place = extract_place(text_resume)
                    fullname = extract_person_full_name(text_resume)
                    cert = extract_institute_organization_certification(text_resume)       
                    #details = resume_details(text_resume)                
                    #matchPercentage = count_vectorizer_text([text_resume, job])
                    #job_terms = extract_terms_by_job(job)
                    #matchkeywords = match_terms_by_resumen_and_job(text_resume, job_terms)
                    process_data.append({#"file_name":file.filename, 
                                    #"match_percentage":matchPercentage,
                                    #"job_term":job_terms,
                                    #"matchkeywords_job_cv":matchkeywords,
                                    #"cv_details": details,
                                    #"Text":text_resume,
                                    "Date":date,
                                    "Place":place,
                                    "Full name":fullname,
                                    "Institute and Certification":cert})
                    success = True
                
            else:                
                errors['message'] = 'File {} : type is not allowed'.format(file.filename)

        if success and errors:
            errors['message'] = 'File(s) successfully uploaded'
            return _500(errors)
        if success:
            return _201({'message' : 'Files successfully uploaded', "job":job, "process":process_data})
        else:
            return _500(errors)

@app.route('/test',  methods=['GET', 'POST'])
def add_income():
    logging.info('Process init')
    if request.method:
        print("got request method POST {} ".format(request.method))
    if request.is_json:
        data = request.get_json()
    print(f"response ->{request.is_json}")
    return jsonify({"data": data}), 200

@app.route('/read',  methods=['GET', 'POST'])
def read():
    resp = extract_text_from_image()
    return jsonify({"data": resp}), 200

@app.route('/verticales',  methods=['GET', 'POST'])
def test():
    cl = request.form.get('client')
    result = "No Vertical found"
    for key in VERTICALES_DB:
        for client in VERTICALES_DB[key].get('client'):
            if client.lower().find(cl.lower()) > -1:
                result = key
    return jsonify({"data": result}), 200

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True, port=8083)
    