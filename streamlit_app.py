# pip install pycryptodome
import urllib.parse
from datetime import datetime
from glob import glob
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import google_auth_httplib2
import httplib2
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import HttpRequest


QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection_2"

SCOPE = "https://www.googleapis.com/auth/spreadsheets"
SHEET_ID = "16V9me0ByryDXxsupaVGlzB1fvWsQZ8EDw8CaO94AfiU"
SHEET_NAME = "sheet1"

@st.cache_resource(ttl=1)
def connect_to_gsheet():
    #st.write("Secrets structure:", st.secrets.keys())
    #st.write("Connections structure:", st.secrets.get("connections", {}).keys())
    # Create a connection object
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["connections"]["gcs"], scopes=[SCOPE]
    )

    # Create a new Http() object for every request
    def build_request(http, *args, **kwargs):
        new_http = google_auth_httplib2.AuthorizedHttp(
            credentials, http=httplib2.Http()
        )

        return HttpRequest(new_http, *args, **kwargs)

    authorized_http = google_auth_httplib2.AuthorizedHttp(
        credentials, http=httplib2.Http()
    )

    service = build("sheets", "v4", requestBuilder=build_request, http=authorized_http)
    #gsheet_connector = service.spreadsheets()
    return service.spreadsheets() 
    #return gsheet_connector

def add_row_to_gsheet(gsheet_connector, row):
    try:
        sheet_metadata = gsheet_connector.get(spreadsheetId=SHEET_ID).execute()
        sheets = sheet_metadata.get('sheets', [])
        sheet_names = [sheet['properties']['title'] for sheet in sheets]
        
        if SHEET_NAME not in sheet_names:
            st.error(f"Sheet '{SHEET_NAME}' not found in the spreadsheet.")
            return

        _, content = row
        
        # 質問と回答を分離して整形
        lines = content.split('\n')
        formatted_row = [''] * 6  # 6列分の空の要素を用意
        
        for i in range(0, min(len(lines), 15), 5):  # 最大3つの質問に対応（各質問は5行）
            question = lines[i].strip()
            options = f"a) {lines[i + 1].strip()}\nb) {lines[i + 2].strip()}\nc) {lines[i + 3].strip()}"
            full_question = f"{question}\n{options}"
            answer = lines[i + 4].strip()
            
            col_index = i // 5 * 2
            formatted_row[col_index] = full_question
            formatted_row[col_index + 1] = answer

        encoded_sheet_name = urllib.parse.quote(SHEET_NAME)
        range_spec = f"{encoded_sheet_name}!A1:F1"
        
        result = gsheet_connector.values().append(
            spreadsheetId=SHEET_ID,
            range=range_spec,
            body=dict(values=[formatted_row]),
            valueInputOption="USER_ENTERED",
        ).execute()
        
        st.success("Data successfully added to Google Sheets")
    except Exception as e:
         st.error(f"Error in add_row_to_gsheet: {str(e)}")
         st.exception(e)

def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",
        page_icon="🤗"
    )
    st.sidebar.title("Nav")
    st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
    elif model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
    else:
        st.session_state.model_name = "gpt-4"
    
    # 300: 本文以外の指示のトークン数 (以下同じ)
    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)


def get_pdf_text():
    uploaded_file = st.file_uploader(
        label='Upload your PDF here😇',
        type='pdf'
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-ada-002",
            # 適切な chunk size は質問対象のPDFによって変わるため調整が必要
            # 大きくしすぎると質問回答時に色々な箇所の情報を参照することができない
            # 逆に小さすぎると一つのchunkに十分なサイズの文脈が入らない
            chunk_size=500,
            chunk_overlap=0,
        )
        return text_splitter.split_text(text)
    else:
        return None


def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)

    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        # コレクションが存在しない場合、新しく作成します
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')

    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME, 
        embeddings=OpenAIEmbeddings()
    )


def build_vector_store(pdf_text):
    qdrant = load_qdrant()
    qdrant.add_texts(pdf_text)

    # 以下のようにもできる。この場合は毎回ベクトルDBが初期化される
    # LangChain の Document Loader を利用した場合は `from_documents` にする
    # Qdrant.from_texts(
    #     pdf_text,
    #     OpenAIEmbeddings(),
    #     path="./local_qdrant",
    #     collection_name="my_documents",
    # )


def build_qa_model(llm):
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        # "mmr",  "similarity_score_threshold" などもある
        search_type="similarity",
        # 文書を何個取得するか (default: 4)
        search_kwargs={"k":10}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )


def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload")
    container = st.container()
    with container:
        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("Loading PDF ..."):
                build_vector_store(pdf_text)


def ask(qa, query):
    with get_openai_callback() as cb:
        # query / result / source_documents
        answer = qa(query)

    return answer, cb.total_cost


def page_ask_my_pdf(gsheet_connector):
    st.title("Ask My PDF(s)")

    llm = select_model()
    container = st.container()
    response_container = st.container()

    with container:
        query = st.text_input("Query: ", key="input")
        if not query:
            answer = None
        else:
            qa = build_qa_model(llm)
            if qa:
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = ask(qa, query)
                st.session_state.costs.append(cost)
            else:
                answer = None

        if answer:
            with response_container:
                st.markdown("## Answer")
                result = answer.get('result', '')  # 'result' キーの値を取得
                st.write(result)
            
            # Google Sheets に追加するデータを準備
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row_data = [current_time, result]
    
            add_row_to_gsheet(gsheet_connector, row_data)


def main():
    init_page()
    gsheet_connector = connect_to_gsheet()

    selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask My PDF(s)"])
    if selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()
    elif selection == "Ask My PDF(s)":
        page_ask_my_pdf(gsheet_connector)

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == '__main__':
    main()