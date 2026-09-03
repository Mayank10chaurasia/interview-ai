from pydantic import BaseModel, Field

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_core.output_parsers import (
    JsonOutputParser
)

from workflow.services.LLM import llm

from workflow.mcp_server.client import call_tool_sync

class EmailOutput(BaseModel):

    subject: str = Field(
        description="Email subject"
    )

    body: str = Field(
        description="Email body"
    )


parser = JsonOutputParser(
    pydantic_object=EmailOutput
)


prompt = ChatPromptTemplate.from_template("""
You are an HR recruiter.

Candidate Name:
{name}

Resume Score:
{score}

Write a professional email informing the candidate
that they have successfully passed the resume
screening and are invited to schedule an interview.

Return only JSON.

{format_instruction}
""")


chain = (
    prompt.partial(
        format_instruction=
        parser.get_format_instructions()
    )
    | llm
    | parser
)

def email_agent(state):

    candidate = state["candidate"]

    result = chain.invoke({
        "name": candidate["name"],
        "score": state["score"],
    })

    status = call_tool_sync("send_email", {
        "account_name": "interviewai",
        "recipients": [candidate["email"]],
        "subject": result["subject"],
        "body": result["body"]
    })

    state["email_subject"] = result["subject"]
    state["email_body"] = result["body"]
    state["email_status"] = status

    print(status)

    return state