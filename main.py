import streamlit as st
from langchain_core.messages.chat import ChatMessage
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain import hub
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tempfile
import os
import hashlib
import fitz  # PyMuPDF
import pandas as pd
from io import StringIO

# 환경변수 설정 및 API 키 관리
def load_config():
    """환경별 호환성을 고려한 설정 로딩"""
    
    # 로컬 개발 환경에서는 .env 파일 로드 시도
    try:
        load_dotenv()
    except:
        pass  # 배포 환경에서는 dotenv가 없을 수 있음
    
    config = {}
    
    # OpenAI API Key
    config['openai_key'] = (
        st.secrets.get("api_keys", {}).get("openai") or  # secrets.toml 섹션 방식
        st.secrets.get("OPENAI_API_KEY") or              # secrets.toml 루트 방식
        os.getenv("OPENAI_API_KEY")                      # 환경변수
    )
    
    # Anthropic API Key
    config['anthropic_key'] = (
        st.secrets.get("api_keys", {}).get("anthropic") or
        st.secrets.get("ANTHROPIC_API_KEY") or
        os.getenv("ANTHROPIC_API_KEY")
    )
    
    # LangSmith 설정 (선택사항)
    config['langsmith_key'] = (
        st.secrets.get("langsmith", {}).get("api_key") or
        st.secrets.get("LANGSMITH_API_KEY") or
        os.getenv("LANGSMITH_API_KEY")
    )
    
    config['langsmith_project'] = (
        st.secrets.get("langsmith", {}).get("project") or
        st.secrets.get("LANGSMITH_PROJECT") or
        os.getenv("LANGSMITH_PROJECT")
    )
    
    return config

def check_api_keys():
    """API 키 설정 상태 확인"""
    config = load_config()
    
    missing_keys = []
    
    if not config['openai_key']:
        missing_keys.append("OpenAI API Key")
    
    if not config['anthropic_key']:
        missing_keys.append("Anthropic API Key")
    
    if missing_keys:
        st.error(f"⚠️ 다음 API 키가 설정되지 않았습니다: {', '.join(missing_keys)}")
        st.info("💡 secrets.toml 파일이나 환경변수를 확인해주세요.")
        with st.expander("🔧 설정 방법"):
            st.code("""
# .streamlit/secrets.toml 파일 생성:
[api_keys]
openai = "your-openai-key"
anthropic = "your-anthropic-key"
""", language="toml")
        return False
    
    return True

# 전역 설정 로드
APP_CONFIG = load_config()

st.set_page_config(page_title="규정 챗봇")
st.title("팜소프트 cGMP 규정 챗봇")
st.caption("cGMP 규정에 대해서 자세히 알려드립니다.")

# API 키 확인 - 앱 시작 시 체크
if not check_api_keys():
    st.stop()  # API 키가 없으면 앱 실행 중단

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# 데이터베이스 초기화를 세션 상태로 관리
@st.cache_resource
def initialize_database():
    """데이터베이스를 초기화하는 함수 (캐시된 리소스로 관리)"""
    try:
        # OpenAI에서 제공하는 Embedding Model을 활용해서 `chunk`를 vector화
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=APP_CONFIG['openai_key']  # API 키 명시적 전달
        )
        
        # 배포 환경을 고려한 저장 디렉토리 설정
        if os.path.exists("./chroma"):
            persist_directory = "./chroma"
        else:
            # 배포환경용 - 임시 디렉토리 사용
            persist_directory = tempfile.mkdtemp(prefix="chroma_")
            st.info(f"임시 저장소 사용: {persist_directory}")
        
        # 데이터를 저장할 때 
        database = Chroma(
            collection_name="chroma-cGMP",
            persist_directory=persist_directory,
            embedding_function=embedding,
        )
        return database
    except Exception as e:
        st.error(f"데이터베이스 초기화 실패: {str(e)}")
        return None

# 데이터베이스 초기화
if "database" not in st.session_state or st.session_state["database"] is None:
    st.session_state["database"] = initialize_database()

# 사이드바 생성
with st.sidebar:
    st.header("📚 문서 관리")
    
    # API 키 상태 표시
    with st.expander("🔑 API 설정 상태"):
        if APP_CONFIG['openai_key']:
            st.success("✅ OpenAI API Key 설정됨")
        else:
            st.error("❌ OpenAI API Key 없음")
            
        if APP_CONFIG['anthropic_key']:
            st.success("✅ Anthropic API Key 설정됨")
        else:
            st.error("❌ Anthropic API Key 없음")
            
        if APP_CONFIG['langsmith_key']:
            st.info(f"📊 LangSmith 프로젝트: {APP_CONFIG['langsmith_project']}")
    
    # 저장된 문서 목록 표시
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = set()
    
    if st.session_state["processed_files"]:
        st.subheader("💾 저장된 문서")
        for filename in st.session_state["processed_files"]:
            st.write(f"✅ {filename}")
        
        # 데이터베이스 상태 확인 버튼
        if st.button("🔍 DB 상태 확인"):
            try:
                if st.session_state["database"]:
                    total_docs = st.session_state["database"].get()
                    st.info(f"총 저장된 청크: {len(total_docs['documents'])}개")
                    
                    # 파일별 청크 수 표시
                    file_counts = {}
                    for metadata in total_docs['metadatas']:
                        if metadata and 'source' in metadata:
                            filename = metadata['source']
                            file_counts[filename] = file_counts.get(filename, 0) + 1
                    
                    st.write("📊 파일별 청크 수:")
                    for filename, count in file_counts.items():
                        st.write(f"  • {filename}: {count}개")
                        
                else:
                    st.error("데이터베이스가 초기화되지 않았습니다.")
                    
            except Exception as e:
                st.error(f"DB 상태 확인 실패: {str(e)}")
    else:
        st.info("아직 업로드된 문서가 없습니다.")
    
    st.divider()
    
    # 초기화 버튼 생성
    clear_btn = st.button("💬 대화 초기화")
    
    # 데이터베이스 초기화 버튼
    if st.button("🗑️ 전체 데이터베이스 초기화", type="secondary"):
        try:
            # 방법 1: 세션 상태에서 데이터베이스 제거
            if "database" in st.session_state:
                del st.session_state["database"]
            
            # 방법 2: 캐시된 리소스 클리어
            initialize_database.clear()
            
            # 방법 3: 물리적 파일 삭제 (로컬 환경인 경우)
            import shutil
            if os.path.exists("./chroma"):
                shutil.rmtree("./chroma")
            
            # 처리된 파일 목록 초기화
            st.session_state["processed_files"] = set()
            st.session_state["messages"] = []
            
            st.success("데이터베이스가 완전히 초기화되었습니다.")
            st.info("페이지를 새로고침하거나 새 문서를 업로드해주세요.")
            
        except Exception as e:
            st.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
            # 오류가 발생해도 세션 상태는 초기화
            if "database" in st.session_state:
                del st.session_state["database"]
            st.session_state["processed_files"] = set()
            st.session_state["messages"] = []
            st.info("세션 상태는 초기화되었습니다. 페이지를 새로고침해주세요.")

def get_file_hash(file_content):
    """파일 내용의 해시값을 생성합니다."""
    return hashlib.md5(file_content).hexdigest()

def is_file_already_processed(filename, file_hash):
    """파일이 이미 처리되었는지 확인합니다."""
    try:
        if not st.session_state.get("database"):
            return False
            
        # 데이터베이스에서 해당 파일의 문서들을 검색
        existing_docs = st.session_state["database"].get(
            where={"source": filename}
        )
        
        # 같은 파일명과 해시값을 가진 문서가 있는지 확인
        if existing_docs['documents']:
            for metadata in existing_docs['metadatas']:
                if metadata and metadata.get('file_hash') == file_hash:
                    return True
        return False
    except Exception as e:
        st.warning(f"중복 확인 중 오류: {str(e)}")
        return False

def extract_pdf_content_advanced(file_path):
    """PyMuPDF를 사용해서 PDF에서 텍스트, 표, 구조 정보를 추출합니다."""
    doc = fitz.open(file_path)
    documents = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 기본 텍스트 추출
        text = page.get_text()
        
        # 표 추출 시도
        tables = page.find_tables()
        table_content = ""
        
        if tables:
            for table_num, table in enumerate(tables):
                try:
                    # 표 데이터를 추출하고 문자열로 변환
                    table_data = table.extract()
                    if table_data:
                        # 표를 마크다운 형식으로 변환
                        table_text = f"\n\n[표 {table_num + 1}]\n"
                        for row in table_data:
                            # None 값을 빈 문자열로 변환
                            clean_row = [str(cell) if cell is not None else "" for cell in row]
                            table_text += "| " + " | ".join(clean_row) + " |\n"
                        table_content += table_text
                except Exception as e:
                    st.warning(f"표 추출 중 오류 (페이지 {page_num + 1}): {str(e)}")
        
        # 이미지 정보 추출
        image_list = page.get_images()
        image_content = ""
        if image_list:
            image_content = f"\n\n[이 페이지에는 {len(image_list)}개의 이미지가 포함되어 있습니다]\n"
        
        # 텍스트 블록 정보 (서식, 위치 등)
        blocks = page.get_text("dict")
        structured_text = ""
        
        for block in blocks["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        # 폰트 크기와 스타일 정보 포함
                        font_size = span.get("size", 12)
                        font_flags = span.get("flags", 0)
                        text_content = span.get("text", "")
                        
                        # 제목이나 중요한 텍스트 식별 (큰 폰트나 볼드)
                        if font_size > 14 or font_flags & 2**4:  # 볼드 체크
                            structured_text += f"\n### {text_content}\n"
                        else:
                            structured_text += text_content
        
        # 모든 내용 결합
        combined_content = f"""
페이지 {page_num + 1} 내용:

{text}

{table_content}

{image_content}

구조화된 텍스트:
{structured_text}
        """.strip()
        
        # Document 객체 생성
        doc_obj = Document(
            page_content=combined_content,
            metadata={
                "page": page_num + 1,
                "has_tables": len(tables) > 0,
                "table_count": len(tables),
                "image_count": len(image_list),
                "total_pages": len(doc)
            }
        )
        documents.append(doc_obj)
    
    doc.close()
    return documents

def process_pdf_file(uploaded_file):
    """PDF 파일을 처리하고 Chroma DB에 저장합니다."""
    try:
        # 데이터베이스 확인
        if not st.session_state.get("database"):
            return f"❌ '{uploaded_file.name}': 데이터베이스가 초기화되지 않았습니다."
        
        # 파일 내용 읽기
        file_content = uploaded_file.read()
        file_hash = get_file_hash(file_content)
        
        # 중복 체크
        if is_file_already_processed(uploaded_file.name, file_hash):
            return f"⚠️ '{uploaded_file.name}'은 이미 처리된 파일입니다."
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # 고급 PDF 추출 시도
            try:
                documents = extract_pdf_content_advanced(tmp_file_path)
                extraction_method = "고급 추출 (표, 이미지, 구조 포함)"
            except Exception as e:
                # 기본 PDF 로더로 폴백
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
                extraction_method = "기본 추출"
            
            if not documents:
                return f"❌ '{uploaded_file.name}': PDF에서 텍스트를 추출할 수 없습니다."
            
            # 텍스트 분할 (더 큰 청크 크기로 표와 구조 보존)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # 표와 구조를 보존하기 위해 더 큰 청크
                chunk_overlap=300,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            
            # 메타데이터에 파일 정보 추가
            for i, split in enumerate(splits):
                original_metadata = split.metadata.copy()
                split.metadata = {
                    'source': uploaded_file.name,
                    'file_hash': file_hash,
                    'upload_time': str(st.session_state.get('current_time', '')),
                    'page': original_metadata.get('page', 1),
                    'chunk_id': f"{uploaded_file.name}_chunk_{i+1}",
                    'has_tables': original_metadata.get('has_tables', False),
                    'table_count': original_metadata.get('table_count', 0),
                    'image_count': original_metadata.get('image_count', 0),
                    'extraction_method': extraction_method
                }
            
            # Chroma DB에 추가
            st.session_state["database"].add_documents(splits)
            
            # 처리된 파일 목록에 추가
            st.session_state["processed_files"].add(uploaded_file.name)
            
            # 상세 정보 포함한 성공 메시지
            table_info = ""
            image_info = ""
            total_tables = sum(doc.metadata.get('table_count', 0) for doc in documents)
            total_images = sum(doc.metadata.get('image_count', 0) for doc in documents)
            
            if total_tables > 0:
                table_info = f", 표 {total_tables}개"
            if total_images > 0:
                image_info = f", 이미지 {total_images}개"
            
            return f"✅ '{uploaded_file.name}': {len(splits)}개 청크로 처리 완료 ({extraction_method}{table_info}{image_info})"
            
        finally:
            # 임시 파일 삭제
            os.unlink(tmp_file_path)
            
    except Exception as e:
        return f"❌ '{uploaded_file.name}': 처리 중 오류 발생 - {str(e)}"

# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

def get_ai_message(user_input):
    # 데이터베이스 확인
    if "database" not in st.session_state or st.session_state["database"] is None:
        return {
            "answer": "데이터베이스가 초기화되지 않았습니다. 페이지를 새로고침하거나 문서를 업로드해주세요.",
            "sources": []
        }
    
    # Claude 모델 초기화 (API 키 명시적 전달)
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=APP_CONFIG['anthropic_key']
    )
    
    try:
        # 검색된 문서들 가져오기 (더 많은 문서 검색)
        retriever = st.session_state["database"].as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # 5개 문서로 증가
        )
        docs = retriever.get_relevant_documents(user_input)
    except Exception as e:
        return {
            "answer": f"문서 검색 중 오류가 발생했습니다: {str(e)}\n\n데이터베이스를 초기화하고 다시 시도해주세요.",
            "sources": []
        }
    
    if not docs:
        return {
            "answer": "죄송합니다. 업로드된 문서에서 관련 정보를 찾을 수 없습니다. 문서가 올바르게 업로드되었는지 확인해주세요.",
            "sources": []
        }
    
    # 컨텍스트 생성
    context = "\n\n".join([f"[문서: {doc.metadata.get('source', '알 수 없음')}]\n{doc.page_content}" for doc in docs])
    
    # 출처 정보 수집 (중복 제거 및 정리)
    sources = []
    for doc in docs:
        if hasattr(doc, 'metadata') and doc.metadata:
            source_name = doc.metadata.get('source', '알 수 없는 출처')
            page_num = doc.metadata.get('page', '')
            
            # 출처 정보 포맷팅
            if page_num and str(page_num).isdigit():
                source_info = f"{source_name} (페이지 {page_num})"
            else:
                source_info = source_name
            
            if source_info not in sources:  # 중복 제거
                sources.append(source_info)
    
    # 프롬프트 템플릿 생성
    prompt_template = """
    당신은 cGMP 규정 전문가입니다. 다음 문서들을 바탕으로 질문에 정확하고 상세하게 답변해주세요.

    문서 내용:
    {context}

    질문: {question}

    답변 시 다음 사항을 준수해주세요:
    1. 제공된 문서 내용만을 바탕으로 답변하세요
    2. 구체적이고 실용적인 정보를 포함하세요
    3. 단계별 절차가 있다면 순서대로 설명하세요
    4. 문서에 없는 내용은 추측하지 마세요

    답변:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # LLM에 질문
    formatted_prompt = prompt.format(context=context, question=user_input)
    response = llm.invoke(formatted_prompt)
    
    # 응답과 출처 정보를 딕셔너리로 반환
    return {
        "answer": response.content if hasattr(response, 'content') else str(response),
        "sources": sources
    }

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 파일 업로드 섹션 (챗 입력 위에)
st.subheader("📎 문서 업로드")
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_files = st.file_uploader(
        "PDF 파일을 업로드해서 챗봇에 학습시키세요",
        type=['pdf'],
        accept_multiple_files=True,
        help="여러 PDF 파일을 동시에 업로드할 수 있습니다.",
        key="main_uploader"
    )

with col2:
    if uploaded_files:
        process_btn = st.button("📥 문서 처리하기", type="primary", key="main_process")

# 파일 처리 로직
if uploaded_files and 'process_btn' in locals() and process_btn:
    import datetime
    st.session_state['current_time'] = str(datetime.datetime.now())
    
    with st.spinner("📄 PDF 파일들을 처리하고 있습니다..."):
        results = []
        for uploaded_file in uploaded_files:
            result = process_pdf_file(uploaded_file)
            results.append(result)
        
        # 결과 표시
        for result in results:
            if result.startswith("✅"):
                st.success(result)
            elif result.startswith("⚠️"):
                st.warning(result)
            else:
                st.error(result)
        
        # 처리 완료 메시지 (애니메이션 제거)
        st.info("📄 문서 처리가 완료되었습니다!")

st.divider()

# 사용자의 입력
user_input = st.chat_input("cGMP 규정 관련하여 궁금한 내용을 말씀해 주세요.")

# 만약에 사용자 입력이 들어오면...
if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)

    # AI 응답
    with st.spinner("답변을 생성하고 있습니다..."):
        try:
            ai_response = get_ai_message(user_input)
            
            # 답변과 출처 분리
            if isinstance(ai_response, dict):
                answer = ai_response.get("answer", "답변을 생성할 수 없습니다.")
                sources = ai_response.get("sources", [])
                
                # AI 답변 표시
                st.chat_message("ai").write(answer)
                
                # 출처 정보 표시
                if sources:
                    with st.expander("📚 참고 출처"):
                        for i, source in enumerate(sources, 1):
                            st.write(f"{i}. {source}")
                
                # 대화기록에는 답변만 저장
                display_message = answer
                if sources:
                    display_message += f"\n\n**참고 출처:**\n" + "\n".join([f"- {source}" for source in sources])
                
            else:
                # 예상치 못한 응답 형태인 경우
                display_message = str(ai_response)
                st.chat_message("ai").write(display_message)
            
            # 대화기록을 저장한다.
            add_message("user", user_input)
            add_message("ai", display_message)
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            st.info("API 키 설정과 Chroma 데이터베이스 상태를 확인해주세요.")