import azure.functions as func
from datetime import datetime, timezone, timedelta
import json
import logging
import pymongo
import uuid
import pytz
import docx
import fitz
import io
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import AzureOpenAI
from openai import OpenAI
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import pandas as pd

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

def default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()

CONNECTION_STRING = 'mongodb+srv://rorymclean:Aderas!123@onemoretime.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000'
DB_NAME = "alldocs"
try:
    client = pymongo.MongoClient(CONNECTION_STRING)
    try:
        client.server_info()  # validate connection string
    except (
        pymongo.errors.OperationFailure,
        pymongo.errors.ConnectionFailure,
        pymongo.errors.ExecutionTimeout,
    ) as err:
        sys.exit("Can't connect:" + str(err))
        # return func.HttpResponse("Unable to connect"+str(err),status_code=503)
except Exception as err:
    sys.exit("Error:" + str(err))
    # return func.HttpResponse("Error: "+str(err),status_code=503)

db = client[DB_NAME]

COLLECTION_NAMES = ['sessions','logs','documents','embeddings','sharepoint']
sessions  = db[COLLECTION_NAMES[0]]
logs      = db[COLLECTION_NAMES[1]]
documents = db[COLLECTION_NAMES[2]]
chunks    = db[COLLECTION_NAMES[3]]
sharepoint = db[COLLECTION_NAMES[4]]

# openai.api_key = '1e8470d9073047df996d72388f9d2882'
# openai.azure_endpoint = 'https://dfc-dev-instance.openai.azure.com/'
# openai.api_type = "azure"
azureclient = AzureOpenAI(
  azure_endpoint = "https://dfc-dev-instance.openai.azure.com/", #os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key = '1e8470d9073047df996d72388f9d2882', #os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2024-02-01",
)

def get_embedding(text, model="small_vector"):
    text = text.replace("\n", " ")
    response = azureclient.embeddings.create(input=[text],model=model )
    myresponse = json.loads(response.json())
    return myresponse['data'][0]['embedding']
    # return emb_client.embeddings.create(input = [text], model=model).data[0].embedding

def get_llm_response(myprompt, model="35_model"):
    stream = azureclient.chat.completions.create(
        model=model, # model = "deployment_name".
        messages=myprompt, #[{"role": "user", "content": "Explain the Balls Bluff battle in two paragraphs"}],
        stream=True
    )

    myresults = ""
    for chunk in stream:
        if chunk.choices is not None:
            if len(chunk.choices) > 0:
                if chunk.choices[0].delta.content is not None:
                    myresults = myresults + chunk.choices[0].delta.content
                    streamresponse.markdown(myresults)

    return myresults

def chunk_add_doc(mysession, mysource, mydata):
    # newkey = doc_add(mysession, mysource)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 5000,
        chunk_overlap  = 0,
        length_function = len,)
    texts = text_splitter.create_documents([mydata])

    data = [{'Text': item.page_content} for item in texts]
    df = pd.DataFrame(data, columns=['Text'])
    df['id'] = range(len(df))

    records = df.to_dict("records")
    pagecount = 0
    for doc in records:
        pagecount += 1
        myembedding = get_embedding(doc['Text'])
        mydict = { "mysession":mysession, "sourceName": mysource, "text": doc['Text'], "vectors": myembedding, "chunk":pagecount, "creationdate": datetime.now(), "expiration": datetime.now(timezone.utc)+ timedelta(hours=12) }
        x = chunks.insert_one(mydict)
    mydict = { "mysession":mysession, "sourceName": mysource, "creationdate": datetime.now(), "expiration": datetime.now(timezone.utc)+ timedelta(hours=12) }
    x = documents.insert_one(mydict)      
        
    return len(df)  

############### Sessions 
@app.route(route="sessionget")
def session_query(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Start Session Query.')
    # Get the query parameters
    try:
        req_body = req.get_json()
    except:
        req_body = {}
    # myuser        
    myuser = req.params.get('myuser')
    if not myuser:
        myuser = req_body.get('myuser')
    if not myuser:
        return func.HttpResponse("No user ID provided",status_code=400)
    # mytpe
    mytype = req.params.get('mytype')
    if not mytype:
        mytype = req_body.get('mytype')
    if not mytype:
        mytype = 'Private'
    logging.info('User ID: '+myuser+" / Type: "+mytype)

    if myuser == "*":
        tempresults = sessions.find().sort('user')
    else:
        tempresults = sessions.find({"$and": [{'user': myuser}, {'type': mytype}]}).sort('user')

    mydocuments = [{k: v for k, v in doc.items() if k != '_id'} for doc in tempresults]
    logging.info("Number of Records: "+str(len(mydocuments)))
    json_string = json.dumps(mydocuments, default=default_serializer)
    return func.HttpResponse(json_string, status_code=200)

@app.route(route="session_add")
def session_add(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Start Session Add.')
    # Get the query parameters
    try:
        req_body = req.get_json()
    except:
        req_body = {}
    # myuser 
    myuser = req.params.get('myuser')
    if not myuser:
        myuser = req_body.get('myuser')
    if not myuser:
        return func.HttpResponse("No user ID provided",status_code=400)
    # mytype
    mytype = req.params.get('mytype')
    if not mytype:
        mytype = req_body.get('mytype')
    if not mytype:
        mytype = 'Private'
    logging.info('User ID: '+myuser+" / Type: "+mytype)

    mysession = str(uuid.uuid4())
    myexpiration = datetime.now(timezone.utc)+ timedelta(hours=12)
    mycreation = datetime.now()
    mydict = { "mysession":mysession, "user": myuser, "type": mytype, "creationdate": mycreation, "expiration": myexpiration }
    myid = sessions.insert_one(mydict)

    logging.info("ID: "+str(myid))
    return func.HttpResponse(mysession, status_code=200)

################ Documents
@app.route(route="doc_get")
def doc_query(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Start Document Query.')

    # Get the query parameters
    mysession = req.params.get('mysession')
    if not mysession:
        try:
            req_body = req.get_json()
            mysession = req_body.get('mysession')
        except ValueError:
            return func.HttpResponse("No Session ID provided",status_code=400)
    logging.info('Session ID: '+mysession)

    if mysession == "*":
        tempresults = documents.find().sort('sourceName')
    else:
        tempresults = documents.find({'mysession': mysession}).sort('sourceName')
    logging.info(tempresults)
    mydocuments = [{k: v for k, v in doc.items() if k != '_id'} for doc in tempresults]
    logging.info("Number of Records: "+str(len(mydocuments)))
    json_string = json.dumps(mydocuments, default=default_serializer)
    return func.HttpResponse(json_string, status_code=200)

@app.route(route="doc_add")
def doc_add(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Start Log Add.')
    # Get the query parameters
    try:
        req_body = req.get_json()
    except:
        req_body = {}
    mysession = req.params.get('mysession')
    if not mysession:
        mysession = req_body.get('mysession')
    if not mysession:
        return func.HttpResponse("No Session ID Provided",status_code=400)

    mysource = req.params.get('mysource')
    if not mysource:
        mysource = req_body.get('mysource')
    if not mysource:
        return func.HttpResponse("No Source Provided",status_code=400)

    logging.info('Session ID: '+mysession+' Source: '+mysource)

    mydict = { "mysession":mysession, "sourceName": mysource, "creationdate": datetime.now(), "expiration": datetime.now(timezone.utc)+ timedelta(hours=12) }
    mykey = documents.insert_one(mydict)
    logging.info("ID: "+str(mykey))
    return func.HttpResponse(str(mykey), status_code=200)

################ Logs
@app.route(route="log_get")
def log_query(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Start Log Query.')
    # Get the query parameters
    try:
        req_body = req.get_json()
    except:
        req_body = {}
    mysession = req.params.get('mysession')
    if not mysession:
        mysession = req_body.get('mysession')
    if not mysession:
        return func.HttpResponse("No Session ID provided",status_code=400)
    logging.info('Session ID: '+mysession)

    if mysession == "*":
        tempresults = logs.find().sort('creationdate')
    else:
        tempresults = logs.find({'mysession': mysession}).sort('creationdate')
    logging.info(tempresults)
    mydocuments = [{k: v for k, v in doc.items() if k != '_id'} for doc in tempresults]
    logging.info("Number of Records: "+str(len(mydocuments)))
    json_string = json.dumps(mydocuments, default=default_serializer)
    return func.HttpResponse(json_string, status_code=200)

@app.route(route="log_add")
def log_add(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Start Log Add.')
    # Get the query parameters
    try:
        req_body = req.get_json()
    except:
        req_body = {}
    mysession = req.params.get('mysession')
    if not mysession:
        mysession = req_body.get('mysession')
    if not mysession:
        return func.HttpResponse("No Session ID Provided",status_code=400)

    myprompt = req.params.get('myprompt')
    if not myprompt:
        myprompt = req_body.get('myprompt')
    if not myprompt:
        return func.HttpResponse("No Prompt Provided",status_code=400)

    myresponse = req.params.get('myresponse')
    if not myresponse:
        myresponse = req_body.get('myresponse')
    if not myresponse:
        return func.HttpResponse("No Response Provided",status_code=400)
    logging.info('Session ID: '+mysession)

    mydict = { "mysession":mysession, "prompt": myprompt, "response": myresponse, "creationdate": datetime.now(), "expiration": datetime.now(timezone.utc)+ timedelta(hours=12) }
    mykey = logs.insert_one(mydict)
    logging.info("ID: "+str(mykey))
    return func.HttpResponse(str(mykey), status_code=200)

################ Chunks
@app.route(route="chunk_get")
def chunk_get(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Start Chunk Get.')

    # Get the query parameters
    mysession = req.params.get('mysession')
    if not mysession:
        try:
            req_body = req.get_json()
            mysession = req_body.get('mysession')
        except ValueError:
            return func.HttpResponse("No Session ID provided",status_code=400)
    logging.info('Session ID: '+mysession)

    if mysession == "*":
        tempresults = chunks.find().sort('sourceName')
    else:
        tempresults = chunks.find({'mysession': mysession}).sort('sourceName')
    logging.info(tempresults)
    mydocuments = [{k: v for k, v in doc.items() if k != '_id'} for doc in tempresults]
    logging.info("Number of Records: "+str(len(mydocuments)))
    json_string = json.dumps(mydocuments, default=default_serializer)
    return func.HttpResponse(json_string, status_code=200)

@app.route(route="chunk_search")
def chunk_search(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Start Chunk Search.')
    # Get the query parameters
    try:
        req_body = req.get_json()
    except:
        req_body = {}    
    mysession = req.params.get('mysession')
    if not mysession:
        mysession = req_body.get('mysession')
    if not mysession:
        return func.HttpResponse("No Session ID provided",status_code=400)
    myquery = req.params.get('myquery')
    if not myquery:
        myquery = req_body.get('myquery')
    if not myquery:
        return func.HttpResponse("No Query provided",status_code=400)
    logging.info(myquery)
    kvalue = req.params.get('kvalue')
    if not kvalue:
        kvalue = req_body.get('kvalue')
    if not kvalue:
        kvalue = 5
    
    logging.info('Session ID: '+mysession+" - "+str(kvalue))
    
    embedded_query = get_embedding(myquery)
    pipeline = [
    {"$search": {
        "cosmosSearch": {
            "vector": embedded_query,
            "path": "vectors",
            "k": 5000,
            # "efSearch": 40
            },
        "returnStoredSource": True }},
        {
            "$project": { "similarityScore": {
                "$meta": "searchScore" },
                "document" : "$$ROOT"
                }
        }
    ]

    cursor = chunks.aggregate(pipeline)

    all_df = pd.DataFrame(list(cursor))
    # pd.set_option('display.max_columns', None)
    # logging.info(all_df.head(5))
    # logging.info("All: "+str(len(all_df)))
    if len(all_df) > 0:
        doc_df = pd.DataFrame(all_df['document'].tolist())
        all_df = pd.concat([all_df[['similarityScore']], doc_df], axis=1)

        my_df = all_df[all_df['mysession'] == mysession]
        logging.info("Filtered: "+str(len(my_df)))
        my_df.sort_values(by='similarityScore', ascending=False, inplace=True)
        myoutput = my_df['text'].head(kvalue).str.cat(sep=' \n\n')
        return func.HttpResponse(myoutput, status_code=200)
    else:
        return func.HttpResponse("", status_code=200)

@app.route(route="chunk_public_search")
def chunk_public_search(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Start Public Chunk Search.')
    # Get the query parameters
    try:
        req_body = req.get_json()
    except:
        req_body = {}    
    mysession = req.params.get('mysession')
    if not mysession:
        mysession = req_body.get('mysession')
    if not mysession:
        return func.HttpResponse("No Session ID provided",status_code=400)
    myquery = req.params.get('myquery')
    if not myquery:
        myquery = req_body.get('myquery')
    if not myquery:
        return func.HttpResponse("No Query provided",status_code=400)
    kvalue = req.params.get('kvalue')
    if not kvalue:
        kvalue = req_body.get('kvalue')
    if not kvalue:
        kvalue = 10
    
    logging.info('Session ID: '+mysession+" - "+str(kvalue))

    embedded_query = get_embedding(myquery)
    pipeline = [
    {"$search": {
        "cosmosSearch": {
            "vector": embedded_query,
            "path": "vectors",
            "k": kvalue,
            # "efSearch": 40
            },
        "returnStoredSource": True }},
        {
            "$project": { "similarityScore": {
                "$meta": "searchScore" },
                "document" : "$$ROOT"
                }
        }
    ]

    cursor = sharepoint.aggregate(pipeline)
    all_df = pd.DataFrame(list(cursor))
    doc_df = pd.DataFrame(all_df['document'].tolist())
    all_df = pd.concat([all_df[['similarityScore']], doc_df], axis=1)
    
    myoutput = all_df['text'].head(kvalue).str.cat(sep=' \n\n')
    return func.HttpResponse(myoutput, status_code=200)

@app.route(route="file_add")
def file_add(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Start File Add.')
    logging.info('*****')
    mysession = req.params.get('mysession')
    for input_file in req.files.values():
        filename = input_file.filename
        file_ext = input_file.filename.split('.')[-1].lower()
        contents = input_file.stream.read()
        logging.info('Filename: %s' % filename)
        logging.info('Extension: %s' % file_ext)
        logging.info("Session: %s" % mysession)
        # logging.info('Contents:')
        # logging.info(contents)
    logging.info('*****')

    stored_filename = filename
    chacters_to_remove = "-|_/\*"
    for char in chacters_to_remove:
        stored_filename = stored_filename.replace(char, '')
    mydata_all = ""
    if file_ext in ["txt","csv","text"]:
        mydata_all = contents.decode('utf-8')
    elif file_ext == "docx":
        doc = docx.Document(io.BytesIO(contents))
        all_paras = doc.paragraphs
        this_doc = ''
        for para in all_paras:
            this_doc = this_doc+para.text + "\n"
        mydata_all = mydata_all + this_doc + '\n'  
    elif file_ext == "pdf":
        doc = fitz.open(stream=contents)
        this_doc = ''
        for page in doc:
            this_doc = this_doc+page.get_text() + " \n"
        mydata_all = mydata_all + this_doc + '\n' 
    elif file_ext == 'xlsx':
        for z in range(10):
            try:
                myexcel = pd.read_excel(contents, sheet_name=z) 
                for index, row in myexcel.iterrows():
                    rowtext = ""
                    for k in myexcel.keys():
                        rowtext = rowtext + str(row[k]) + " | "
                mydata_all = mydata_all + rowtext + "\n"
            except:
                pass

    totchars = len(mydata_all)
    if totchars > 0:
        chunk_cnt = chunk_add_doc(mysession, stored_filename, mydata_all)
 
    return func.HttpResponse("finished", status_code=200)