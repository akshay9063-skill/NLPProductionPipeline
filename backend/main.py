import logging
import time

from fastapi import Depends, FastAPI, Request

from backend.schemas import (
    QARequest,
    QAResponse,
    TriageRequest,
    TriageResponse,
)
from backend.qa import answer_question
from backend.triage import triage_ticket
from backend.security import get_current_role
from monitoring.metrics import record_request, get_metrics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("nlp_production_pipeline")


app = FastAPI(
    title="NLP Production Pipeline",
    version="0.1.0",
    description=(
        "Documentation question answering and "
        "support ticket triage API with monitoring."
    ),
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request metadata and record operational metrics."""
    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        logger.error(
            "Request failed: method=%s path=%s error_type=%s",
            request.method,
            request.url.path,
            type(exc).__name__,
        )
        raise
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000

        record_request(
            request.method,
            request.url.path,
            status_code,
            duration_ms,
        )

        logger.info(
            "HTTP method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            status_code,
            duration_ms,
        )


@app.get("/health")
def health():
    """Return basic service health information."""
    return {
        "status": "ok",
        "service": "NLP Production Pipeline",
        "version": "0.1.0",
    }


@app.get("/metrics")
def metrics(role: str = Depends(get_current_role)):
    """Return operational metrics to authenticated callers."""
    return get_metrics()


@app.post("/qa", response_model=QAResponse)
def qa(
    request: QARequest,
    role: str = Depends(get_current_role),
):
    """Answer questions using documents authorized for the caller."""
    return answer_question(
        request.question,
        role=role,
    )


@app.post("/triage", response_model=TriageResponse)
def triage(
    request: TriageRequest,
    role: str = Depends(get_current_role),
):
    """Classify a support ticket for an authenticated caller."""
    return triage_ticket(request.ticket)
