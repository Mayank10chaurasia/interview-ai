import base64
import os
import tempfile

from dotenv import load_dotenv
from langsmith import traceable

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_core.output_parsers import (
    JsonOutputParser
)

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_ollama import OllamaEmbeddings

from langchain_community.vectorstores import FAISS

from workflow.states.candidate import (
    CandidateProfile
)

from workflow.services.LLM import llm


load_dotenv()


# =========================================
# LOAD RESUME
# =========================================

@traceable(name="load_resume")
def load_resume(state):

    pdf_source = state["pdf_path"]

    temp_path = None

    try:

        # =====================================
        # BASE64 PDF
        # =====================================

        if pdf_source.startswith(
            "data:application/pdf;base64,"
        ):

            base64_data = pdf_source.split(
                ",",
                1
            )[1]

            pdf_bytes = base64.b64decode(
                base64_data
            )

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            )

            temp_file.write(pdf_bytes)

            temp_file.close()

            temp_path = temp_file.name

            pdf_path = temp_path

        else:

            # Normal local PDF path
            pdf_path = pdf_source


        # =====================================
        # LOAD PDF
        # =====================================

        loader = PyPDFLoader(
            pdf_path
        )

        docs = loader.load()


        text = ""

        for page in docs:

            text += (
                page.page_content
                + "\n"
            )


        return {
            "resume_text": text
        }


    finally:

        # Delete temporary PDF
        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            os.remove(temp_path)


# =========================================
# CHUNKING
# =========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


@traceable(name="chunking")
def chunk_resume(state):

    chunks = splitter.create_documents(
        [state["resume_text"]]
    )

    return {
        "chunks": chunks
    }


# =========================================
# EMBEDDINGS
# =========================================

embedding = OllamaEmbeddings(
    model="nomic-embed-text"
)


# =========================================
# RETRIEVAL
# =========================================

@traceable(name="retrieve")
def retrieve_resume(state):
    

    db = FAISS.from_documents(
        state["chunks"],
        embedding
    )

    retriever = db.as_retriever(
        search_kwargs={
            "k": 3
        }
    ) 
    jd_for_search = state["jd"][:3000]

    docs = retriever.invoke(
        jd_for_search
    )
    MAX_CONTEXT_CHARS = 4000

      
    context = "\n".join(
        doc.page_content
        for doc in docs
    )
    context = context[:MAX_CONTEXT_CHARS]  

    return {
        "retrieved_context": context
    }


# =========================================
# CANDIDATE PARSER
# =========================================

candidate_parser = JsonOutputParser(
    pydantic_object=CandidateProfile
)


candidate_prompt = ChatPromptTemplate.from_template("""
You are an expert Resume Parser.

Extract candidate information from the resume.

Rules:

- Retrieve name and email.
- Do not hallucinate.
- If information is missing return "".
- If technologies are missing return [].
- Return ONLY valid JSON.

Resume:

{resume}

{format_instructions}
""")


candidate_chain = (
    candidate_prompt.partial(
        format_instructions=
        candidate_parser.get_format_instructions()
    )
    | llm
    | candidate_parser
)

@traceable(name="extract_candidate")
def extract_candidate(state):

    # Prevent very large resume context
    resume_text = state["resume_text"][:8000]

    candidate = candidate_chain.invoke({
        "resume": resume_text
    })

    return {
        "candidate": candidate
    }