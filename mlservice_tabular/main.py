"""
Main FastAPI application module.

This module provides the main FastAPI application with Swagger UI documentation 
and dynamic route registration capabilities. The API documentation is available 
at /docs endpoint.
"""

import argparse
from mlservice.main import setup_routes, app

import uvicorn
def main():
    """Run the FastAPI application."""
    parser = argparse.ArgumentParser(description="Run the ML Service API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--external-routines", nargs="+", help="List of external routine modules to import")
    args = parser.parse_args()
    setup_routes(['mlservice_tabular'])
    setup_routes(args.external_routines)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
