"""
Main FastAPI application.
"""
import logging
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from src.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create FastAPI app
app = FastAPI(
    title="NLP Document Processing API",
    description="API for processing documents with natural language processing",
    version="1.0.0",
    docs_url=None,  # Disable default docs to customize
    redoc_url=None  # Disable default redoc to customize
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions and return a standardized response.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "detail": getattr(exc, "detail_message", None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle general exceptions and return a standardized response.
    """
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if os.environ.get("ENVIRONMENT") == "development" else None
        }
    )

# Root redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

# Custom docs endpoint
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI docs."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="NLP Document Processing API",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
    )

# Custom OpenAPI schema
@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """Custom OpenAPI schema."""
    return get_openapi(
        title="NLP Document Processing API",
        version="1.0.0",
        description="API for processing documents with natural language processing",
        routes=app.routes,
    )

# Health check endpoint
@app.get("/health", include_in_schema=False)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# For development only
if __name__ == "__main__":
    import uvicorn
    
    # Get environment variables
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("ENVIRONMENT") == "development"
    
    uvicorn.run("src.api.main:app", host=host, port=port, reload=reload)