from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback

from pipeline import run_pipeline, run_pipeline_live
from file_utils import read_uploaded_bytes

app = FastAPI(title="Hallucination Verification Lab")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/check")
async def api_check(request: Request):
    try:
        form = await request.form()

        source_text = str(form.get("source_text", "") or "")
        llm_response = str(form.get("llm_response", "") or "")
        model = str(form.get("model", "llama3") or "llama3")

        source_file = form.get("source_file")
        if source_file is not None and getattr(source_file, "filename", ""):
            content = await source_file.read()
            extracted = read_uploaded_bytes(source_file.filename, content)
            if extracted:
                source_text = extracted

        response_file = form.get("response_file")
        if response_file is not None and getattr(response_file, "filename", ""):
            content = await response_file.read()
            extracted = read_uploaded_bytes(response_file.filename, content)
            if extracted:
                llm_response = extracted

        if not llm_response.strip():
            return JSONResponse(status_code=400, content={"error": "LLM response is empty."})
        if not source_text.strip():
            return JSONResponse(status_code=400, content={"error": "Source document is empty."})

        output = run_pipeline(source_text, llm_response, model=model)
        return output

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/check-live")
async def api_check_live(request: Request):
    try:
        form = await request.form()

        llm_response = str(form.get("llm_response", "") or "")
        model = str(form.get("model", "llama3") or "llama3")

        response_file = form.get("response_file")
        if response_file is not None and getattr(response_file, "filename", ""):
            content = await response_file.read()
            extracted = read_uploaded_bytes(response_file.filename, content)
            if extracted:
                llm_response = extracted

        if not llm_response.strip():
            return JSONResponse(status_code=400, content={"error": "LLM response is empty."})

        output = run_pipeline_live(llm_response, model=model)
        return output

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


app.mount("/", StaticFiles(directory="static", html=True), name="static")
