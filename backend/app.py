"""
AutoExtract Backend Application

A Flask-based web application that provides APIs for automatic data extraction
from various types of graphs and charts using deep learning models.

This application offers:
- Bar graph analysis with YOLO-based object detection
- OCR text extraction from graph elements
- Interactive coordinate and value editing
- RESTful API endpoints for frontend integration

Author: Mohan Yang
Repository: https://github.com/HenryYang03/AutoExtract.git
"""

from flask import Flask, abort, jsonify, send_from_directory
import os

from config import UPLOAD_DIR, SECRET_KEY, HOST, PORT, DEBUG
from model_manager import model_manager
from routes import handle_bar_analyzer, handle_update_values, handle_update_box_coordinates, handle_uploaded_file, handle_calculate_heights, handle_update_bar_names, handle_update_box_category, handle_add_new_box, handle_remove_box
from extract_routes import handle_extract, handle_export, handle_job_file


def create_app() -> Flask:
    """
    Create and configure the Flask application.
    
    This factory function creates a Flask app instance, configures it,
    registers routes, and ensures required directories exist.
    
    Returns:
        Flask: Configured Flask application instance
        
    Example:
        >>> app = create_app()
        >>> app.run()
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Configure app
    app.secret_key = SECRET_KEY
    app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
    app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024   # whole papers can be large PDFs
    
    # Ensure upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Register routes
    register_routes(app)

    # Production: serve Vite build from FRONTEND_DIST (same origin as /api)
    register_frontend_spa(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    return app


def register_frontend_spa(app: Flask) -> None:
    """Serve the built React app when FRONTEND_DIST points at `vite build` output."""
    dist_dir = os.environ.get("FRONTEND_DIST", "").strip()
    if not dist_dir or not os.path.isdir(dist_dir):
        return

    dist_real = os.path.realpath(dist_dir)

    def _under_dist(candidate: str) -> bool:
        cr = os.path.realpath(candidate)
        return cr == dist_real or cr.startswith(dist_real + os.sep)

    @app.route("/assets/<path:filename>")
    def spa_assets(filename):
        assets = os.path.join(dist_dir, "assets")
        if not os.path.isdir(assets):
            abort(404)
        target = os.path.join(assets, filename)
        if not _under_dist(target) or not os.path.isfile(target):
            abort(404)
        return send_from_directory(assets, filename)

    @app.route("/", defaults={"spa_path": ""})
    @app.route("/<path:spa_path>")
    def spa_catch_all(spa_path):
        if spa_path.startswith("api/"):
            abort(404)
        if spa_path.startswith("static/uploads/"):
            abort(404)
        if spa_path:
            target = os.path.normpath(os.path.join(dist_dir, spa_path))
            if _under_dist(target) and os.path.isfile(target):
                return send_from_directory(dist_dir, spa_path)
        return send_from_directory(dist_dir, "index.html")


def register_routes(app: Flask) -> None:
    """
    Register all API routes with the Flask application.
    
    Args:
        app (Flask): Flask application instance to register routes with
        
    Routes:
        POST /api/bar_analyzer: Process uploaded images for graph analysis
        POST /api/update_values: Update origin and ymax values
        GET /static/uploads/<filename>: Serve uploaded files
    """
    # API routes
    app.add_url_rule(
        '/api/bar_analyzer',
        'bar_analyzer',
        handle_bar_analyzer,
        methods=['POST']
    )
    
    app.add_url_rule(
        '/api/update_values',
        'update_values',
        handle_update_values,
        methods=['POST']
    )
    
    app.add_url_rule(
        '/api/update_box_coordinates',
        'update_box_coordinates',
        handle_update_box_coordinates,
        methods=['POST']
    )
    
    app.add_url_rule(
        '/api/calculate_heights',
        'calculate_heights',
        handle_calculate_heights,
        methods=['POST']
    )
    
    app.add_url_rule(
        '/api/update_bar_names',
        'update_bar_names',
        handle_update_bar_names,
        methods=['POST']
    )
    
    app.add_url_rule(
        '/api/update_box_category',
        'update_box_category',
        handle_update_box_category,
        methods=['POST']
    )
    
    app.add_url_rule(
        '/api/add_new_box',
        'add_new_box',
        handle_add_new_box,
        methods=['POST']
    )
    
    app.add_url_rule(
        '/api/remove_box',
        'remove_box',
        handle_remove_box,
        methods=['POST']
    )
    
    # Batch figure extraction (PDFs or images -> every bar-chart panel)
    app.add_url_rule('/api/extract', 'extract', handle_extract, methods=['POST'])
    app.add_url_rule('/api/export', 'export', handle_export, methods=['POST'])

    # File serving routes
    app.add_url_rule(
        '/static/uploads/<filename>',
        'uploaded_file',
        handle_uploaded_file
    )
    app.add_url_rule('/static/jobs/<job>/<filename>', 'job_file', handle_job_file)


def register_error_handlers(app: Flask) -> None:
    """
    Register error handlers for the Flask application.
    
    Args:
        app (Flask): Flask application instance to register error handlers with
    """
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors."""
        return jsonify({'error': 'Resource not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server errors."""
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        """Handle unhandled exceptions."""
        return jsonify({'error': 'An unexpected error occurred'}), 500


def main():
    """
    Main entry point for the application.
    
    Creates the Flask app, checks model readiness, and starts the server.
    """
    # Create and configure app
    app = create_app()
    
    # Check if model is ready
    if not model_manager.is_ready():
        print("Warning: BarGraphAnalyzer model failed to initialize!")
        print("The application may not function properly.")
    else:
        print("✓ BarGraphAnalyzer model initialized successfully")
    
    # Start the server
    print(f"Starting AutoExtract backend server on {HOST}:{PORT}")
    print(f"Debug mode: {'ON' if DEBUG else 'OFF'}")
    
    app.run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == "__main__":
    main()
