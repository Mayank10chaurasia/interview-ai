from workflow.interview_graph.interview_agent import (
    start_interview,
    submit_answer,
)
import requests

from fastapi import FastAPI, HTTPException, Request

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from workflow.resume_screeining_agent.resume_screening_agent import (
    run_resume_graph,
)

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Interview AI Service",
    version="1.0.0",
)


# =========================================================
# CORS
# React :5173 -> FastAPI :8000
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================================================
# EXPRESS BACKEND
# =========================================================

EXPRESS_API = "http://localhost:5000/api/resume"


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Interview AI Python service running",
        "status": "ok",
    }


@app.get("/health")
def health():

    return {
        "ok": True,
        "service": "interview-ai",
    }


# =========================================================
# RESUME SCREENING
# =========================================================

@app.post(
    "/process-resume/{application_id}"
)
def process_resume(
    application_id: str,
):

    print(
        f"\nReceived application: {application_id}"
    )

    # -----------------------------------------------------
    # GET APPLICATION FROM EXPRESS
    # -----------------------------------------------------

    try:

        response = requests.get(
            f"{EXPRESS_API}/{application_id}",
            timeout=30,
        )

    except requests.RequestException as error:

        print(
            "Express connection error:",
            error,
        )

        raise HTTPException(
            status_code=503,
            detail="Could not connect to Express API",
        )


    if response.status_code != 200:

        raise HTTPException(
            status_code=response.status_code,
            detail="Could not load application",
        )


    application = response.json()


    print(
        "Resume exists:",
        bool(
            application.get("resumeUrl")
        ),
    )

    print(
        "JD exists:",
        bool(
            application.get("jobDescription")
        ),
    )


    # -----------------------------------------------------
    # EXTRACT DATA
    # -----------------------------------------------------

    resume_url = application.get(
        "resumeUrl"
    )

    job_description = application.get(
        "jobDescription"
    )


    if not resume_url:

        raise HTTPException(
            status_code=400,
            detail="Resume not found",
        )


    if not job_description:

        raise HTTPException(
            status_code=400,
            detail="Job description not found",
        )


    # -----------------------------------------------------
    # INITIAL LANGGRAPH STATE
    # -----------------------------------------------------

    state = {
        "pdf_path": resume_url,
        "jd": job_description,
    }


    # -----------------------------------------------------
    # RUN RESUME GRAPH
    #
    # IMPORTANT:
    # application_id = LangGraph thread_id
    # -----------------------------------------------------

    try:

        result = run_resume_graph(
            state,
            application_id,
        )

    except Exception as error:

        print(
            "Resume LangGraph error:",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


    if not result:

        raise HTTPException(
            status_code=500,
            detail="Resume graph returned no result",
        )


    print(
        "\n========== RESUME ANALYSIS =========="
    )

    print(result)


    # -----------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------

    analysis = result.get(
        "analysis",
        {},
    )


    score_data = {

        "matchPercent": int(
            result.get(
                "score",
                0,
            )
        ),

        "confidence": 0,

        "strengths": analysis.get(
            "strengths",
            [],
        ),

        "weaknesses": analysis.get(
            "weaknesses",
            [],
        ),

        "missingSkills": analysis.get(
            "missing_skills",
            [],
        ),

        "recommendation": result.get(
            "decision",
            "",
        ),

        "decision": result.get(
            "decision",
            "",
        ),
    }


    # -----------------------------------------------------
    # SAVE SCORE TO EXPRESS / MONGODB
    # -----------------------------------------------------

    try:

        score_response = requests.post(
            f"{EXPRESS_API}/{application_id}/score",
            json=score_data,
            timeout=30,
        )


        if score_response.status_code >= 400:

            print(
                "Could not save AI score:",
                score_response.text,
            )

    except requests.RequestException as error:

        # Don't destroy the completed AI analysis merely
        # because Mongo/Express update failed.

        print(
            "Could not send score to Express:",
            error,
        )


    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {

        "message":
            "Resume analysis completed",

        "applicationId":
            application_id,

        "score":
            result.get("score"),

        "analysis":
            analysis,

        "decision":
            result.get("decision"),
    }


# =========================================================
# START AI INTERVIEW
# =========================================================

@app.post(
    "/interview/{application_id}/start"
)
def start_ai_interview(
    application_id: str,
):

    print(
        f"\nStarting interview: {application_id}"
    )


    try:

        # application_id is also the
        # LangGraph thread_id.

        result = start_interview(
            application_id
        )

    except HTTPException:
        raise

    except Exception as error:

        print(
            "Start interview error:",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


    if not result:

        raise HTTPException(
            status_code=500,
            detail="Interview graph returned no result",
        )


    print(
        "Interview started:",
        result,
    )


    return {

        "completed": result.get(
            "completed",
            False,
        ),

        "question": result.get(
            "question"
        ),

        "question_count": result.get(
            "question_count",
            0,
        ),

        "result": result.get("result"),
    }


# =========================================================
# SUBMIT CANDIDATE ANSWER
# =========================================================

@app.post(
    "/interview/{application_id}/answer"
)
async def answer_interview(
    application_id: str,
    request: Request,
):
    print(f"\nAnswer received for: {application_id}")

    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        payload = await request.json()
        transcript = payload.get("transcript") if isinstance(payload, dict) else None
    else:
        form = await request.form()
        transcript = form.get("transcript")

    if not isinstance(transcript, str):
        raise HTTPException(
            status_code=422,
            detail="A transcript string is required",
        )

    transcript = transcript.strip()
    print("Candidate transcript:", transcript)

    if not transcript:
        raise HTTPException(
            status_code=400,
            detail="Transcript is required",
        )

    try:
        result = submit_answer(application_id, transcript)
    except Exception as error:
        print("Interview graph error:", error)
        raise HTTPException(status_code=500, detail=str(error))

    if not result:
        raise HTTPException(
            status_code=500,
            detail="Interview graph returned no result",
        )

    response = {
        "transcript": transcript,
        "question": result.get("question"),
        "question_count": result.get("question_count", 0),
        "completed": result.get("completed", False),
        "result": result.get("result") if result.get("completed") else None,
    }

    print("Sending to React:", response)
    return response