"""
Configuration module for AutoExtract backend application.

This module contains all configuration settings, paths, and constants
used throughout the application.
"""

import os


# Base directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload configuration
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model configuration
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')
CLASS_NAMES = [
    'label', 'ymax', 'origin', 'yaxis', 'bar', 'uptail', 
    'legend', 'legend_group', 'xaxis', 'x_group'
]

# Flask configuration
SECRET_KEY = "b'\xca\xa4\xf2\x80!\xfe\x85\xba\xd7\xcf\xe7\xc9\xf1)I\xac\x10Y5M\x95\xed\xfb\xc4'"
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "9000"))
DEBUG = os.environ.get("DEBUG", "true").lower() in ("1", "true", "yes")

# OCR configuration
TESSERACT_CMD = 'tesseract' 