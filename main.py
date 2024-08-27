import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# 외부 라이브러리와 모듈을 불러옵니다.
# Streamlit: 웹 애플리케이션 인터페이스를 만들기 위해 사용합니다.
# os: 파일 경로 작업을 위해 사용합니다.
# langchain 및 관련 모듈: 문서 로드, 텍스트 분할, 임베딩 생성, 검색기 구성 등을 위한 라이브러리입니다.

# 환경 변수 로드
os.environ["GOOGLE_API_KEY"] = st.secrets["general"]["google_api_key"]
os.environ["UPSTAGE_API_KEY"] = st.secrets["general"]["upstage_api_key"]

# 프로젝트 로깅 설정
from langchain_teddynote import logging
logging.langsmith("CC_REGULATION_Chatbot")

# Streamlit 웹 앱의 제목 설정
st.title("춘천문화원 제규정 GPT 💬")

# 사용자의 대화 상태를 초기화합니다.
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # 사용자의 대화 메시지를 저장하기 위한 리스트
    st.session_state["db_initialized"] = False  # 데이터베이스 초기화 상태를 추적하는 플래그

# 사이드바에서 초기화 버튼을 생성합니다.
with st.sidebar:
    clear_btn = st.button("대화 초기화")

# 챗봇이 참조할 PDF 파일의 경로를 설정합니다.
pdf_files = [
    "data/regulations.pdf"
]

# 텍스트 분할기를 설정합니다. 긴 텍스트를 작은 청크로 나누는 역할을 합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)

# 임베딩 모델을 설정합니다. 텍스트를 벡터로 변환하는 데 사용됩니다.
embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")

# 벡터 저장소(FAISS)의 경로와 이름을 설정합니다.
faiss_db_path = "faiss_db"
faiss_index_name = "faiss_index"

# 벡터 저장소를 초기화하고 구축하는 함수입니다.
def initialize_db():
    if not os.path.exists(faiss_db_path):  # 저장소가 없는 경우
        documents = []
        for pdf_file in pdf_files:
            loader = PDFPlumberLoader(pdf_file)  # PDF 파일을 로드합니다.
            split_docs = loader.load_and_split(text_splitter)  # PDF를 작은 청크로 나눕니다.
            documents.extend([Document(page_content=str(page)) for page in split_docs])

        # FAISS DB를 생성하고 로컬에 저장합니다.
        db = FAISS.from_documents(documents=documents, embedding=embeddings)
        db.save_local(folder_path=faiss_db_path, index_name=faiss_index_name)
    else:
        # 기존에 저장된 FAISS DB를 로드합니다.
        db = FAISS.load_local(
            folder_path=faiss_db_path,
            index_name=faiss_index_name,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
    st.session_state["db_initialized"] = True  # 데이터베이스가 초기화됨을 표시합니다.
    return db

# 데이터베이스를 로드하거나 초기화합니다.
if not st.session_state["db_initialized"]:
    loaded_db = initialize_db()  # 데이터베이스를 초기화합니다.
else:
    loaded_db = FAISS.load_local(  # 이미 초기화된 경우 기존 DB를 로드합니다.
        folder_path=faiss_db_path,
        index_name=faiss_index_name,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

# 검색기를 설정합니다. 사용자의 질문에 관련된 문서를 검색하는 데 사용됩니다.
retriever = loaded_db.as_retriever()

# 이전 대화 메시지를 출력하는 함수입니다.
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message["role"]).write(chat_message["content"])

# 새로운 메시지를 추가하는 함수입니다.
def add_message(role, message):
    st.session_state["messages"].append({"role": role, "content": message})

# 질의응답 체인을 생성하는 함수입니다.
def create_chain():
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Answer in Korean.
        Make sure the output is all the same size characters.
        Never enclose your answer in parentheses.
        Be sure to include your source and page numbers in your answer.
        Please follow the rules below when attributing sources.
        - "regulations.pdf" : "춘천문화원 제규정집"
            
        #Example Format:
        (brief summary of the answer)
        (table)
        (detailed answer to the question)
    
        **출처**
        - (page source and page number)

        #Question: 
        {question}

        #Context: 
        {context} 

        #Answer:"""
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# 초기화 버튼이 눌렸을 경우
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록을 출력합니다.
print_messages()

# 사용자 입력을 받습니다.
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 사용자가 질문을 입력했을 경우
if user_input:
    # 사용자 입력을 기록하고 화면에 출력합니다.
    st.chat_message("user").write(user_input)
    add_message("user", user_input)

    # 체인을 생성합니다.
    chain = create_chain()

    # 검색기를 사용하여 관련 문서를 가져옵니다.
    context_docs = retriever.invoke(user_input)

    if context_docs:
        # 관련 문서에서 컨텍스트를 결합합니다.
        context = "\n".join([doc.page_content for doc in context_docs])
        
        # 답변을 생성합니다.
        response = chain.invoke({"question": user_input, "context": context})

        # 답변을 화면에 출력하고 저장합니다.
        st.chat_message("assistant").write(response)
        add_message("assistant", response)
    else:
        # 관련 문서를 찾지 못했을 경우의 기본 답변
        response = "해당 문서에서 답변을 찾을 수 없습니다. 다른 질문을 시도해 주세요."
        st.chat_message("assistant").write(response)
        add_message("assistant", response)