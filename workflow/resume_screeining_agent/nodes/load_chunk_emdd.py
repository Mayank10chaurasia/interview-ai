from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from workflow.states.candidate import CandidateProfile
from workflow.services.LLM import llm

from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

#loading resume
@traceable(name="load_resume")
def load_resume(state):
    loader = PyPDFLoader(state["pdf_path"])
    docs = loader.load()

    text = ""

    for page in docs:
        text += page.page_content + "\n"

    state["resume_text"] = text

    return state

#chunking resume


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
@traceable(name="chunking")
def chunk_resume(state):

    chunks = splitter.create_documents(
        [state["resume_text"]]
    )

    state["chunks"] = chunks

    return state

#emedding resume
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embedding = OllamaEmbeddings(
            model="nomic-embed-text"
        )

@traceable(name="retrive")
def retrieve_resume(state):

    db = FAISS.from_documents(
        state["chunks"],
        embedding
    )

    retriever = db.as_retriever()

    docs = retriever.invoke("summary of resume")

    context = "\n".join(
        doc.page_content
        for doc in docs
    )
    print("============================context==========================")
    print(context)

    state["retrieved_context"] = context

    return state


parser = JsonOutputParser(
    pydantic_object=CandidateProfile
)

prompt = ChatPromptTemplate.from_template("""
You are an expert Resume Parser.

Extract the candidate information from the resume.

Rules:
-retrive name and email
- Do not hallucinate.
- If information is missing return "".
- If technologies are missing return [].
- Return ONLY valid JSON.

Resume:
{resume}

{format_instructions}
""")

chain = (
    prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    | llm
    | parser
)


@traceable(name="extract_candidate")
def extract_candidate(state):

    candidate = chain.invoke({
        "resume": state["resume_text" ]
    })
    

    return {
        "candidate": candidate
    }
 