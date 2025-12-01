"""
Transcripción Asistida - Página de revisión fila por fila
Esta es la versión actual de transcripción asistida con revisión manual
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all necessary modules directly
import streamlit as st
import pandas as pd
import json
import logging
import os
from datetime import datetime
from PIL import Image

from excel_image_extractor import ExcelImageExtractor
from api_extractor import APIExtractor
from image_preprocessor import ImagePreprocessor
from ocr_assistant import OCRAssistant
from config import get_config

# Import all functions from the main module
import assisted_transcription_ui_v2 as main_module

# Set page config
st.set_page_config(
    page_title="Transcripción Asistida",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Run the main function from the module
if __name__ == "__main__":
    main_module.main()
