"""
Assisted Transcription UI v2 - On-Demand Processing with Checkpoints
Streamlit-based interface for human-validated data extraction
NEW: Process images one-by-one, select method per image, auto-save checkpoints
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from PIL import Image
import time
import json
import logging
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from excel_image_extractor import ExcelImageExtractor
from api_extractor import APIExtractor
from image_preprocessor import ImagePreprocessor
from ocr_assistant import OCRAssistant
from config import get_config
from shared_results import SharedResultsManager


# Page configuration
st.set_page_config(
    page_title="Transcripción Asistida Industrial v2",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stTextInput > label {
        font-size: 14px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==================== FORMAT CONVERSION UTILITIES ====================

def convert_procesamiento_to_transcripcion(batch_results: list) -> list:
    """
    Convert Procesamiento Rápido format to Transcripción Asistida format
    
    Procesamiento Rápido format:
    {
        'BarCode': 'ABC123',
        'results_by_type': {
            'PLACA1': {marca, modelo, ...},
            'SCADA1': {codigo_scada_principal, ...}
        },
        'images_processed': [...],
        'timestamp': '...'
    }
    
    Transcripción Asistida format:
    {
        'BarCode': 'ABC123',
        'placa_tecnica_img0': {tipo, marca, modelo, method, ...},
        'codigo_scada_img0': {tipo, codigo_scada_principal, method, ...},
        'metodo_extraccion': 'imported_from_procesamiento_rapido',
        'timestamp': '...'
    }
    """
    transcription_results = []
    
    for result in batch_results:
        barcode = result.get('BarCode')
        if not barcode:
            continue
        
        new_result = {
            'BarCode': barcode,
            'timestamp': result.get('timestamp', datetime.now().isoformat()),
            'source': 'imported_from_procesamiento_rapido',
            'metodo_extraccion': 'imported_from_procesamiento_rapido'  # KEY: This is what display_progress looks for
        }
        
        results_by_type = result.get('results_by_type', {})
        placa_counter = 0
        scada_counter = 0
        
        for tipo_label, tipo_data in results_by_type.items():
            tipo_upper = tipo_label.upper()
            
            # Handle AMBOS (creates both placa and scada entries)
            if 'AMBOS' in tipo_upper or 'BOTH' in tipo_upper:
                # PLACA entry
                placa_key = f"placa_tecnica_img{placa_counter}"
                new_result[placa_key] = {
                    'tipo': 'placa_tecnica',
                    'tipo_label_original': tipo_label,
                    'marca': tipo_data.get('marca'),
                    'modelo': tipo_data.get('modelo'),
                    'numero_serie': tipo_data.get('numero_serie'),
                    'año': tipo_data.get('año'),
                    'potencia': tipo_data.get('potencia'),
                    'voltaje': tipo_data.get('voltaje'),
                    'corriente': tipo_data.get('corriente'),
                    'frecuencia': tipo_data.get('frecuencia'),
                    'rpm': tipo_data.get('rpm'),
                    'frame': tipo_data.get('frame'),
                    'fecha_fabricacion': tipo_data.get('fecha_fabricacion'),
                    'ip_protection': tipo_data.get('ip_protection'),
                    'insulation_class': tipo_data.get('insulation_class'),
                    'observaciones': tipo_data.get('observaciones'),
                    'confidence': tipo_data.get('confidence', 0.0),
                    'method': 'imported_from_procesamiento_rapido',
                    'manual_edit': tipo_data.get('manual_edit', False),
                    'raw_response': tipo_data.get('raw_response')  # Transfer API response
                }
                placa_counter += 1
                
                # SCADA entry
                scada_key = f"codigo_scada_img{scada_counter}"
                new_result[scada_key] = {
                    'tipo': 'codigo_scada',
                    'tipo_label_original': tipo_label,
                    'codigo_scada_principal': tipo_data.get('codigo_scada_principal'),
                    'codigo_scada_alternativo': tipo_data.get('codigo_scada_alternativo'),
                    'confidence': tipo_data.get('confidence', 0.0),
                    'method': 'imported_from_procesamiento_rapido',
                    'manual_edit': tipo_data.get('manual_edit', False),
                    'raw_response': tipo_data.get('raw_response')  # Transfer API response
                }
                scada_counter += 1
            
            # Handle PLACA variants
            elif 'PLACA' in tipo_upper:
                placa_key = f"placa_tecnica_img{placa_counter}"
                new_result[placa_key] = {
                    'tipo': 'placa_tecnica',
                    'tipo_label_original': tipo_label,
                    'marca': tipo_data.get('marca'),
                    'modelo': tipo_data.get('modelo'),
                    'numero_serie': tipo_data.get('numero_serie'),
                    'año': tipo_data.get('año'),
                    'potencia': tipo_data.get('potencia'),
                    'voltaje': tipo_data.get('voltaje'),
                    'corriente': tipo_data.get('corriente'),
                    'frecuencia': tipo_data.get('frecuencia'),
                    'rpm': tipo_data.get('rpm'),
                    'frame': tipo_data.get('frame'),
                    'fecha_fabricacion': tipo_data.get('fecha_fabricacion'),
                    'ip_protection': tipo_data.get('ip_protection'),
                    'insulation_class': tipo_data.get('insulation_class'),
                    'observaciones': tipo_data.get('observaciones'),
                    'confidence': tipo_data.get('confidence', 0.0),
                    'method': 'imported_from_procesamiento_rapido',
                    'manual_edit': tipo_data.get('manual_edit', False),
                    'raw_response': tipo_data.get('raw_response')  # Transfer API response
                }
                placa_counter += 1
            
            # Handle SCADA variants
            elif 'SCADA' in tipo_upper:
                scada_key = f"codigo_scada_img{scada_counter}"
                new_result[scada_key] = {
                    'tipo': 'codigo_scada',
                    'tipo_label_original': tipo_label,
                    'codigo_scada_principal': tipo_data.get('codigo_scada_principal'),
                    'codigo_scada_alternativo': tipo_data.get('codigo_scada_alternativo'),
                    'confidence': tipo_data.get('confidence', 0.0),
                    'method': 'imported_from_procesamiento_rapido',
                    'manual_edit': tipo_data.get('manual_edit', False),
                    'raw_response': tipo_data.get('raw_response')  # Transfer API response
                }
                scada_counter += 1
            
            else:
                logger.warning(f"Unknown tipo_label format: {tipo_label} for barcode {barcode}")
        
        transcription_results.append(new_result)
    
    return transcription_results


def convert_transcripcion_to_procesamiento(transcription_results: list) -> list:
    """
    Convert Transcripción Asistida format back to Procesamiento Rápido format
    For exporting/syncing edited data back
    """
    batch_results = []
    
    for result in transcription_results:
        barcode = result.get('BarCode')
        if not barcode:
            continue
        
        new_result = {
            'BarCode': barcode,
            'timestamp': result.get('timestamp', datetime.now().isoformat()),
            'results_by_type': {},
            'status': 'success'
        }
        
        # Group by image index
        placa_entries = {}
        scada_entries = {}
        
        for key, value in result.items():
            if key.startswith('placa_tecnica_img'):
                img_idx = key.replace('placa_tecnica_img', '')
                placa_entries[img_idx] = value
            elif key.startswith('codigo_scada_img'):
                img_idx = key.replace('codigo_scada_img', '')
                scada_entries[img_idx] = value
        
        # Convert back to PLACA1, PLACA2, etc.
        for idx, data in sorted(placa_entries.items()):
            tipo_label = f"PLACA{int(idx) + 1}"
            new_result['results_by_type'][tipo_label] = {
                'marca': data.get('marca'),
                'modelo': data.get('modelo'),
                'numero_serie': data.get('numero_serie'),
                'año': data.get('año'),
                'potencia': data.get('potencia'),
                'voltaje': data.get('voltaje'),
                'corriente': data.get('corriente'),
                'frecuencia': data.get('frecuencia'),
                'rpm': data.get('rpm'),
                'frame': data.get('frame'),
                'fecha_fabricacion': data.get('fecha_fabricacion'),
                'ip_protection': data.get('ip_protection'),
                'insulation_class': data.get('insulation_class'),
                'observaciones': data.get('observaciones'),
                'confidence': data.get('confidence', 0.0),
                'method': data.get('method', 'manual'),
                'manual_edit': data.get('manual_edit', True)
            }
        
        for idx, data in sorted(scada_entries.items()):
            tipo_label = f"SCADA{int(idx) + 1}"
            new_result['results_by_type'][tipo_label] = {
                'codigo_scada_principal': data.get('codigo_scada_principal'),
                'codigo_scada_alternativo': data.get('codigo_scada_alternativo'),
                'confidence': data.get('confidence', 0.0),
                'method': data.get('method', 'manual'),
                'manual_edit': data.get('manual_edit', True)
            }
        
        batch_results.append(new_result)
    
    return batch_results


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_index = 0
        st.session_state.data_df = None
        st.session_state.results = []
        st.session_state.config = get_config()
        st.session_state.checkpoint_file = None
        st.session_state.start_time = datetime.now()
        st.session_state.times_per_image = []
        st.session_state.current_method = "OCR Local"  # Default for current image
        st.session_state.current_model = None
        st.session_state.preprocessed_cache = {}  # Cache preprocessed images
        
        # Multi-image processing state
        st.session_state.current_image_index = 0  # Which image within the row
        st.session_state.current_row_data = None  # Consolidated data for current row
        st.session_state.images_processed = {}  # Track processed images: {img_idx: {type, method, data}}


# ==================== CHECKPOINT SYSTEM ====================

def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    import numpy as np
    import math
    
    # Handle None first
    if obj is None:
        return None
    
    # Handle numpy arrays and pandas Series BEFORE isna check
    if isinstance(obj, np.ndarray):
        return [convert_to_json_serializable(item) for item in obj.tolist()]
    
    if isinstance(obj, pd.Series):
        return [convert_to_json_serializable(item) for item in obj.tolist()]
    
    # Handle dict BEFORE isna check
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    
    # Handle list/tuple BEFORE isna check
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    
    # Now safe to check pd.isna for scalar values
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        # If pd.isna fails (e.g., on complex objects), continue
        pass
    
    # Handle numeric types
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        # Check for NaN or inf
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    
    if isinstance(obj, float):
        # Check for NaN or inf in regular floats too
        try:
            if math.isnan(obj) or math.isinf(obj):
                return None
        except (ValueError, TypeError):
            pass
        return obj
    
    # Handle strings and other basic types
    if isinstance(obj, (str, int, bool)):
        return obj
    
    # Fallback: try to convert to string
    try:
        return str(obj)
    except:
        return None


def save_checkpoint(data_df: pd.DataFrame, current_index: int, results: list, excel_path: str):
    """Save current progress to checkpoint file"""
    checkpoint_dir = Path(st.session_state.config.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_file = checkpoint_dir / f"checkpoint_{Path(excel_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Safely calculate totals
    try:
        total_images = int(pd.to_numeric(data_df['image_count'], errors='coerce').fillna(0).sum())
    except:
        total_images = 0
    
    # Convert all data to JSON-serializable types
    checkpoint_data = {
        'version': '2.0',
        'excel_path': str(excel_path),
        'current_index': int(current_index),
        'total_rows': int(len(data_df)),
        'total_images': total_images,
        'results': convert_to_json_serializable(results),
        'saved_at': datetime.now().isoformat(),
        'data_summary': {
            'barcodes': [str(x) if not pd.isna(x) else None for x in data_df['BarCode'].tolist()],
            'image_counts': [int(pd.to_numeric(x, errors='coerce')) if not pd.isna(x) else 0 for x in data_df['image_count'].tolist()]
        }
    }
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    st.session_state.checkpoint_file = str(checkpoint_file)
    
    # === SYNC TO SHARED RESULTS ===
    try:
        shared_manager = SharedResultsManager(st.session_state.config.output_dir)
        imported = shared_manager.import_from_transcripcion_asistida(results)
        logger.info(f"✅ Synced {imported} results to shared storage (Transcripción Asistida)")
    except Exception as sync_error:
        logger.warning(f"Failed to sync to shared results: {sync_error}")
    
    return checkpoint_file


def load_checkpoint(checkpoint_path: str):
    """Load progress from checkpoint file"""
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        return checkpoint_data
    except json.JSONDecodeError as e:
        logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
        raise ValueError(f"Checkpoint corrupto o inválido. Error en línea {e.lineno}: {e.msg}")
    except Exception as e:
        logger.error(f"Error reading checkpoint {checkpoint_path}: {e}")
        raise


def list_available_checkpoints():
    """List all available checkpoint files"""
    checkpoint_dir = Path(st.session_state.config.output_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return checkpoints


def save_api_key_to_env(api_key: str) -> bool:
    """
    Save OpenAI API key to .env file
    
    Args:
        api_key: OpenAI API key to save
    
    Returns:
        True if successful, False otherwise
    """
    try:
        env_path = Path(__file__).parent / '.env'
        
        # Read existing .env content
        existing_lines = []
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()
        
        # Update or add OPENAI_API_KEY
        key_found = False
        new_lines = []
        
        for line in existing_lines:
            if line.strip().startswith('OPENAI_API_KEY=') or line.strip().startswith('#OPENAI_API_KEY='):
                new_lines.append(f'OPENAI_API_KEY={api_key}\n')
                key_found = True
            else:
                new_lines.append(line)
        
        # If key wasn't found, add it
        if not key_found:
            if new_lines and not new_lines[-1].endswith('\n'):
                new_lines.append('\n')
            new_lines.append(f'OPENAI_API_KEY={api_key}\n')
        
        # Write back to .env
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        # Update environment variable in current process
        import os
        os.environ['OPENAI_API_KEY'] = api_key
        
        logger.info(f"API key saved to {env_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving API key: {e}")
        return False


def test_openai_api_key(api_key: str) -> tuple[bool, str]:
    """
    Test if OpenAI API key is valid
    
    Args:
        api_key: API key to test
    
    Returns:
        Tuple of (success, message)
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        # Try a simple API call (list models - minimal cost)
        client.models.list()
        
        return True, "✅ API key válida"
        
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return False, "❌ API key inválida"
        elif "rate limit" in error_msg.lower():
            return True, "⚠️ API key válida pero límite alcanzado"
        else:
            return False, f"❌ Error: {error_msg[:100]}"


# ==================== DATA LOADING ====================

def load_excel_data(excel_path: str, output_dir: Path):
    """Load and extract images from Excel"""
    with st.spinner("Cargando rutas de imágenes desde Excel..."):
        extractor = ExcelImageExtractor(output_dir)
        df = extractor.extract_from_excel(
            excel_path, 
            barcode_column='BarCode', 
            image_start_column='BJ',
            use_file_paths=True
        )
        
        df_with_images = df[df['has_images'] == True].reset_index(drop=True)
        
        stats = extractor.get_stats()
        
        if stats['total_images'] == 0 or len(df_with_images) == 0:
            st.error(f"⚠️ No se encontraron imágenes. Verifica que:")
            st.error(f"   - Las celdas contengan rutas a archivos de imágenes")
            st.error(f"   - Las carpetas con las imágenes estén en la misma ubicación que el Excel")
            st.error(f"   - Las rutas sean correctas (ej: 'DESCARGA 20250411_Image/IMG_000127_18057.jpg')")
            st.warning(f"📊 Estadísticas:")
            st.warning(f"   - Filas procesadas: {stats.get('total_rows', 0)}")
            st.warning(f"   - Filas con imágenes: {stats.get('rows_with_images', 0)}")
            st.warning(f"   - Total de imágenes: {stats.get('total_images', 0)}")
            st.warning(f"   - Errores: {stats.get('errors', 0)}")
            return None
        
        st.success(f"✅ Excel cargado: {len(df_with_images)} filas con {stats['total_images']} imágenes")
        return df_with_images


# ==================== ON-DEMAND PREPROCESSING ====================

def preprocess_image_on_demand(image_path: str) -> str:
    """Preprocess single image on-demand (with caching)"""
    # Check cache first
    if image_path in st.session_state.preprocessed_cache:
        return st.session_state.preprocessed_cache[image_path]
    
    # Initialize preprocessor if needed
    if not hasattr(st.session_state, 'preprocessor') or st.session_state.preprocessor is None:
        preprocess_config = st.session_state.config.get('preprocessing', {})
        st.session_state.preprocessor = ImagePreprocessor(preprocess_config)
    
    # Preprocess
    preprocessed_dir = Path(st.session_state.config.output_dir) / "preprocessed"
    preprocessed_dir.mkdir(exist_ok=True)
    
    img_path = Path(image_path)
    preprocessed_path = preprocessed_dir / f"{img_path.stem}_prep.jpg"
    
    if not preprocessed_path.exists():
        with st.spinner(f"Preprocesando {img_path.name}..."):
            try:
                st.session_state.preprocessor.preprocess(str(img_path), str(preprocessed_path))
            except Exception as e:
                st.warning(f"Error preprocesando: {e}")
                preprocessed_path = img_path  # Use original if preprocessing fails
    
    # Cache result
    st.session_state.preprocessed_cache[image_path] = str(preprocessed_path)
    
    return str(preprocessed_path)


# ==================== ON-DEMAND EXTRACTION ====================

def run_extraction_on_image(image_path: str, barcode: str, method: str, model_id: str = None, image_type: str = None):
    """Run extraction (OCR or API) on image with type-specific prompts"""
    # Preprocess image first
    preprocessed_path = preprocess_image_on_demand(image_path)
    
    with st.spinner(f"Ejecutando {method}..."):
        if method == "OCR Local":
            # Initialize OCR if needed
            if not hasattr(st.session_state, 'ocr_assistant') or st.session_state.ocr_assistant is None:
                ocr_config = st.session_state.config.get('local', {}).get('ocr', {})
                st.session_state.ocr_assistant = OCRAssistant(ocr_config)
            
            result = st.session_state.ocr_assistant.extract_from_image(image_path, barcode, preprocessed_path)
        
        elif method == "API OpenAI":
            # Initialize API extractor if needed or if model changed
            if (not hasattr(st.session_state, 'api_extractor') or 
                st.session_state.api_extractor is None or
                (model_id and st.session_state.get('current_api_model') != model_id)):
                
                api_config = st.session_state.config.get('api', {}).copy()
                if model_id:
                    api_config['model'] = model_id
                    st.session_state.current_api_model = model_id
                
                st.session_state.api_extractor = APIExtractor(api_config)
            
            result = st.session_state.api_extractor.extract_from_image_assisted(
                image_path, barcode, preprocessed_path, image_type=image_type
            )
        
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    return result, preprocessed_path


# ==================== UI COMPONENTS ====================

def get_confidence_badge(confidence: float, threshold_high: float = 0.85, threshold_medium: float = 0.60) -> str:
    """Get HTML badge for confidence level"""
    if confidence >= threshold_high:
        level = 'high'
    elif confidence >= threshold_medium:
        level = 'medium'
    else:
        level = 'low'
    
    badges = {
        'high': '<span class="confidence-high">🟢 Alta</span>',
        'medium': '<span class="confidence-medium">🟡 Media</span>',
        'low': '<span class="confidence-low">🔴 Baja</span>'
    }
    
    return f"{badges[level]} ({confidence:.0%})"


def display_image_gallery(image_paths: list, barcode: str, current_image_idx: int):
    """
    Display gallery of images with selection and processing status
    
    Args:
        image_paths: List of image paths
        barcode: BarCode for the row
        current_image_idx: Currently selected image index
    
    Returns:
        Selected image index
    """
    st.subheader(f"📷 Imágenes del Activo: {barcode}")
    st.caption(f"Total: {len(image_paths)} imagen(es)")
    
    # Create thumbnail gallery
    num_cols = min(4, len(image_paths))
    cols = st.columns(num_cols)
    
    selected_idx = current_image_idx
    
    for idx, img_path in enumerate(image_paths):
        col_idx = idx % num_cols
        with cols[col_idx]:
            # Check if image is processed
            is_processed, img_type, img_method = get_image_processing_status(idx)
            
            # Status indicator
            if is_processed:
                if img_type == 'sin_datos':
                    status = "⏭️ Omitida"
                    color = "gray"
                else:
                    status = "✅ Procesada"
                    color = "green"
                type_label = {
                    'placa_tecnica': '📋 Placa',
                    'codigo_scada': '🔢 SCADA',
                    'sin_datos': '❌ Sin datos'
                }.get(img_type, '❓')
                st.markdown(f":{color}[**Imagen {idx + 1}** - {type_label}]")
                st.caption(f"{status} ({img_method})")
            else:
                if idx == current_image_idx:
                    st.markdown(f":blue[**Imagen {idx + 1}** ▶️ Actual]")
                else:
                    st.markdown(f"**Imagen {idx + 1}**")
                st.caption("⏳ Pendiente")
            
            # Show thumbnail
            try:
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
            
            # Selection button
            if st.button(f"Seleccionar", key=f"select_img_{idx}", use_container_width=True):
                selected_idx = idx
    
    return selected_idx


def display_images(image_paths: list, barcode: str):
    """Display images with original/preprocessed toggle"""
    st.subheader("📷 Imágenes")
    
    view_mode = st.radio(
        "Vista:",
        ["Original", "Preprocesada", "Comparación"],
        horizontal=True,
        key="view_mode"
    )
    
    for img_idx, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        
        if view_mode == "Original":
            if img_path.exists():
                st.image(str(img_path), caption=f"Imagen {img_idx + 1} - Original", use_container_width=True)
        
        elif view_mode == "Preprocesada":
            prep_path = st.session_state.preprocessed_cache.get(str(img_path))
            if prep_path and Path(prep_path).exists():
                st.image(prep_path, caption=f"Imagen {img_idx + 1} - Preprocesada", use_container_width=True)
            else:
                # Preprocess on-demand
                prep_path = preprocess_image_on_demand(str(img_path))
                st.image(prep_path, caption=f"Imagen {img_idx + 1} - Preprocesada", use_container_width=True)
        
        elif view_mode == "Comparación":
            col1, col2 = st.columns(2)
            with col1:
                if img_path.exists():
                    st.image(str(img_path), caption="Original", use_container_width=True)
            with col2:
                prep_path = st.session_state.preprocessed_cache.get(str(img_path))
                if not prep_path:
                    prep_path = preprocess_image_on_demand(str(img_path))
                if Path(prep_path).exists():
                    st.image(prep_path, caption="Preprocesada", use_container_width=True)
        
        if len(image_paths) > 1:
            st.markdown("---")


def display_extraction_form(row, extraction_result, method: str):
    """Display data entry form with extraction pre-fill"""
    st.subheader("📝 Datos del Activo")
    
    # Show extraction method and confidence
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Método:** {method}")
        if 'model' in extraction_result:
            st.caption(f"Modelo: {extraction_result['model']}")
    with col2:
        overall_conf = extraction_result.get('overall_confidence', 0.0)
        st.markdown(get_confidence_badge(overall_conf), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # BarCode (read-only)
    barcode = row['BarCode']
    st.text_input("BarCode", value=barcode, disabled=True)
    
    # Tipo de Imagen
    tipo_imagen = extraction_result.get('tipo_imagen', 'desconocido')
    tipo_options = ['placa_tecnica', 'codigo_scada', 'desconocido']
    tipo_index = tipo_options.index(tipo_imagen) if tipo_imagen in tipo_options else 2
    
    form_tipo_imagen = st.selectbox(
        "Tipo de Imagen",
        tipo_options,
        index=tipo_index,
        format_func=lambda x: {
            'placa_tecnica': '📋 Placa Técnica',
            'codigo_scada': '🔢 Código SCADA',
            'desconocido': '❓ Desconocido'
        }[x]
    )
    
    # Get field confidence
    field_conf = extraction_result.get('field_confidence', {})
    
    # Fields based on tipo_imagen
    form_data = {'tipo_imagen': form_tipo_imagen}
    
    if form_tipo_imagen == 'placa_tecnica':
        st.markdown("#### Datos de Placa Técnica")
        
        # Marca
        marca_conf = field_conf.get('marca', 0.0)
        st.markdown(f"**Marca** {get_confidence_badge(marca_conf)}", unsafe_allow_html=True)
        form_data['marca'] = st.text_input(
            "Marca",
            value=extraction_result.get('marca', '') or '',
            key="marca",
            label_visibility="collapsed"
        )
        
        # Modelo
        modelo_conf = field_conf.get('modelo', 0.0)
        st.markdown(f"**Modelo** {get_confidence_badge(modelo_conf)}", unsafe_allow_html=True)
        form_data['modelo'] = st.text_input(
            "Modelo",
            value=extraction_result.get('modelo', '') or '',
            key="modelo",
            label_visibility="collapsed"
        )
        
        # Número de Serie
        sn_conf = field_conf.get('numero_serie', 0.0)
        st.markdown(f"**Número de Serie** {get_confidence_badge(sn_conf)}", unsafe_allow_html=True)
        form_data['numero_serie'] = st.text_input(
            "Número de Serie",
            value=extraction_result.get('numero_serie', '') or '',
            key="numero_serie",
            label_visibility="collapsed"
        )
        
        # Año
        año_conf = field_conf.get('año', 0.0)
        st.markdown(f"**Año** {get_confidence_badge(año_conf)}", unsafe_allow_html=True)
        form_data['año'] = st.number_input(
            "Año",
            min_value=1950,
            max_value=2025,
            value=extraction_result.get('año') or 2020,
            key="año",
            label_visibility="collapsed"
        )
        
        form_data['codigo_scada_principal'] = None
    
    elif form_tipo_imagen == 'codigo_scada':
        st.markdown("#### Código SCADA")
        
        scada_conf = field_conf.get('codigo_scada_principal', 0.0)
        st.markdown(f"**Código SCADA Principal** {get_confidence_badge(scada_conf)}", unsafe_allow_html=True)
        form_data['codigo_scada_principal'] = st.text_input(
            "Código SCADA",
            value=extraction_result.get('codigo_scada_principal', '') or '',
            key="codigo_scada",
            label_visibility="collapsed"
        )
        
        form_data['marca'] = None
        form_data['modelo'] = None
        form_data['numero_serie'] = None
        form_data['año'] = None
    
    else:
        form_data['marca'] = None
        form_data['modelo'] = None
        form_data['numero_serie'] = None
        form_data['año'] = None
        form_data['codigo_scada_principal'] = None
    
    # Notas
    st.markdown("---")
    form_data['notas'] = st.text_area(
        "Notas (opcional)",
        value='',
        key="notas"
    )
    
    # Show extracted text and metadata in expander
    with st.expander("📄 Ver texto extraído completo"):
        processing_time = extraction_result.get('processing_time', 0)
        st.caption(f"⏱️ Tiempo de procesamiento: {processing_time:.2f}s")
        
        texto_completo = extraction_result.get('texto_completo', '')
        if texto_completo:
            st.text_area(
                "Texto completo extraído:",
                value=texto_completo,
                height=200,
                disabled=True,
                label_visibility="collapsed"
            )
        else:
            st.info("No se extrajo texto adicional")
        
        # Show raw API response if available (for debugging)
        if method == "API OpenAI" and 'raw_response' in extraction_result:
            with st.expander("🔍 Ver respuesta raw del API (debug)"):
                st.json(extraction_result.get('raw_response', {}))
    
    return form_data


def display_consolidated_form(row, barcode: str):
    """
    Display consolidated form with data from all processed images
    Shows which image contributed each field
    """
    st.subheader("📝 Datos Consolidados del Activo")
    st.caption("Datos recopilados de todas las imágenes procesadas")
    
    # Get consolidated data
    consolidated = get_consolidated_data()
    
    # BarCode (read-only)
    st.text_input("BarCode", value=barcode, disabled=True)
    
    st.markdown("---")
    st.markdown("#### Datos de Placa Técnica")
    
    # Show which images were processed for placa
    if st.session_state.current_row_data:
        placa_images = [idx for idx, info in st.session_state.current_row_data['images_processed'].items() 
                       if info['type'] == 'placa_tecnica']
        if placa_images:
            st.caption(f"🔍 Datos de imagen(es): {', '.join([str(i+1) for i in placa_images])}")
    
    form_data = {}
    
    # Marca
    form_data['marca'] = st.text_input(
        "Marca",
        value=consolidated.get('marca') or '',
        key=f"marca_consolidated_{barcode}"
    )
    
    # Modelo
    form_data['modelo'] = st.text_input(
        "Modelo",
        value=consolidated.get('modelo') or '',
        key=f"modelo_consolidated_{barcode}"
    )
    
    # Número de Serie
    form_data['numero_serie'] = st.text_input(
        "Número de Serie",
        value=consolidated.get('numero_serie') or '',
        key=f"numero_serie_consolidated_{barcode}"
    )
    
    # Año (handle concatenated values)
    año_value = consolidated.get('año')
    # If concatenated (e.g., "2019 + 2007"), use text_input instead
    if año_value and isinstance(año_value, str) and '+' in str(año_value):
        form_data['año'] = st.text_input(
            "Año (múltiples valores)",
            value=str(año_value),
            key=f"año_consolidated_{barcode}"
        )
    else:
        # Normal numeric input
        try:
            año_int = int(año_value) if año_value else 2020
        except (ValueError, TypeError):
            año_int = 2020
        form_data['año'] = st.number_input(
            "Año",
            min_value=1950,
            max_value=2025,
            value=año_int,
            key=f"año_consolidated_{barcode}"
        )
    
    # Potencia (expanded fields like Procesamiento Rápido)
    col1, col2 = st.columns(2)
    with col1:
        form_data['potencia'] = st.text_input(
            "Potencia (HP/kW)",
            value=consolidated.get('potencia') or '',
            key=f"potencia_consolidated_{barcode}"
        )
    with col2:
        form_data['voltaje'] = st.text_input(
            "Voltaje (V)",
            value=consolidated.get('voltaje') or '',
            key=f"voltaje_consolidated_{barcode}"
        )
    
    col3, col4 = st.columns(2)
    with col3:
        form_data['corriente'] = st.text_input(
            "Corriente (A)",
            value=consolidated.get('corriente') or '',
            key=f"corriente_consolidated_{barcode}"
        )
    with col4:
        form_data['frecuencia'] = st.text_input(
            "Frecuencia (Hz)",
            value=consolidated.get('frecuencia') or '',
            key=f"frecuencia_consolidated_{barcode}"
        )
    
    col5, col6 = st.columns(2)
    with col5:
        form_data['rpm'] = st.text_input(
            "RPM",
            value=consolidated.get('rpm') or '',
            key=f"rpm_consolidated_{barcode}"
        )
    with col6:
        form_data['frame'] = st.text_input(
            "Frame",
            value=consolidated.get('frame') or '',
            key=f"frame_consolidated_{barcode}"
        )
    
    st.markdown("---")
    st.markdown("#### Código SCADA")
    
    # Show which images were processed for SCADA
    if st.session_state.current_row_data:
        scada_images = [idx for idx, info in st.session_state.current_row_data['images_processed'].items() 
                       if info['type'] == 'codigo_scada']
        if scada_images:
            st.caption(f"🔍 Datos de imagen(es): {', '.join([str(i+1) for i in scada_images])}")
    
    # Código SCADA Principal
    form_data['codigo_scada_principal'] = st.text_input(
        "Código SCADA Principal",
        value=consolidated.get('codigo_scada_principal') or '',
        key=f"codigo_scada_consolidated_{barcode}"
    )
    
    # Código SCADA Alternativo
    form_data['codigo_scada_alternativo'] = st.text_input(
        "Código SCADA Alternativo",
        value=consolidated.get('codigo_scada_alternativo') or '',
        key=f"codigo_scada_alt_consolidated_{barcode}"
    )
    
    st.markdown("---")
    
    # Notas
    form_data['notas'] = st.text_area(
        "Notas (opcional)",
        value='',
        key=f"notas_consolidated_{barcode}"
    )
    
    # Show processing summary
    if st.session_state.current_row_data:
        with st.expander("📊 Ver resumen de procesamiento", expanded=False):
            processed_count = len(st.session_state.current_row_data['images_processed'])
            st.write(f"**Imágenes procesadas:** {processed_count}")
            st.markdown("---")
            
            for idx, info in st.session_state.current_row_data['images_processed'].items():
                tipo_label = {
                    'placa_tecnica': '📋 Placa Técnica',
                    'codigo_scada': '🔢 Código SCADA',
                    'sin_datos': '❌ Sin datos'
                }.get(info['type'], '❓')
                
                method_label = info['method'].upper()
                st.write(f"**Imagen {idx + 1}:** {tipo_label} - Método: {method_label}")
                
                # Show API response if available
                if info.get('raw_response'):
                    st.caption(f"🤖 Respuesta de la API:")
                    try:
                        import json
                        # Try to parse and pretty-print JSON
                        if isinstance(info['raw_response'], str):
                            response_data = json.loads(info['raw_response'])
                        else:
                            response_data = info['raw_response']
                        
                        st.json(response_data)
                    except:
                        # If not valid JSON, show as text
                        st.code(str(info['raw_response']), language='text')
                
                st.markdown("---")
            
            # Also show raw_response from imported data if no images were processed
            if processed_count == 0:
                current_barcode = st.session_state.current_row_data.get('barcode')
                if current_barcode:
                    # Look for existing result
                    existing_result = next((r for r in st.session_state.results if r.get('BarCode') == current_barcode), None)
                    
                    if existing_result:
                        st.write("**Datos importados desde Procesamiento Rápido:**")
                        st.markdown("---")
                        
                        # Check all placa_tecnica and codigo_scada entries
                        for key, value in existing_result.items():
                            if key.startswith('placa_tecnica_img') or key.startswith('codigo_scada_img'):
                                tipo_label = '📋 Placa Técnica' if 'placa_tecnica' in key else '🔢 Código SCADA'
                                img_num = key.split('_img')[-1]
                                method = value.get('method', 'unknown').upper()
                                
                                st.write(f"**Imagen {int(img_num) + 1}:** {tipo_label} - Método: {method}")
                                
                                if value.get('raw_response'):
                                    st.caption(f"🤖 Respuesta de la API:")
                                    try:
                                        import json
                                        if isinstance(value['raw_response'], str):
                                            response_data = json.loads(value['raw_response'])
                                        else:
                                            response_data = value['raw_response']
                                        
                                        st.json(response_data)
                                    except:
                                        st.code(str(value['raw_response']), language='text')
                                
                                st.markdown("---")
    
    return form_data


def display_progress(current: int, total: int):
    """Display progress bar and statistics"""
    progress = (current + 1) / total if total > 0 else 0
    st.progress(progress)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Progreso", f"{current + 1}/{total}")
    with col2:
        remaining = total - current - 1
        st.metric("Restantes", remaining)
    with col3:
        completed = len(st.session_state.results)
        st.metric("Completados", completed)
    
    # Method statistics
    if st.session_state.results:
        methods_used = {}
        for result in st.session_state.results:
            method = result.get('metodo_extraccion', 'unknown')
            methods_used[method] = methods_used.get(method, 0) + 1
        
        if methods_used:
            st.markdown("---")
            st.caption("📊 Métodos utilizados:")
            method_cols = st.columns(len(methods_used))
            for idx, (method, count) in enumerate(methods_used.items()):
                with method_cols[idx]:
                    emoji = "🤖" if method == "api" else "🔍" if method == "ocr" else "❓"
                    st.metric(f"{emoji} {method.upper()}", count)


def display_navigation(total: int):
    """Display navigation controls"""
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("⏮️ Primero", use_container_width=True):
            st.session_state.current_index = 0
            st.rerun()
    
    with col2:
        if st.button("◀️ Anterior", use_container_width=True):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()
    
    with col3:
        jump_to = st.number_input(
            "Ir a:",
            min_value=1,
            max_value=total,
            value=st.session_state.current_index + 1,
            key="jump_input"
        )
        if st.button("↗️ Saltar", use_container_width=True):
            st.session_state.current_index = jump_to - 1
            st.rerun()
    
    with col4:
        if st.button("▶️ Siguiente", use_container_width=True):
            if st.session_state.current_index < total - 1:
                st.session_state.current_index += 1
                st.rerun()
    
    with col5:
        if st.button("⏭️ Último", use_container_width=True):
            st.session_state.current_index = total - 1
            st.rerun()


def initialize_row_data(barcode: str):
    """
    Initialize data structure for processing multiple images in a row
    Now pre-loads data from imported results if available
    """
    logger.info(f"🔧 initialize_row_data called for barcode: {barcode}")
    logger.info(f"   Total results in session: {len(st.session_state.results)}")
    
    # Check if this barcode already has imported data
    existing_result = next((r for r in st.session_state.results if r.get('BarCode') == barcode), None)
    logger.info(f"   Found existing_result: {existing_result is not None}")
    
    if existing_result:
        logger.info(f"   Existing result keys: {list(existing_result.keys())}")
    
    # Initialize consolidated data
    consolidated = {
        'marca': None,
        'modelo': None,
        'numero_serie': None,
        'año': None,
        'potencia': None,
        'voltaje': None,
        'corriente': None,
        'frecuencia': None,
        'rpm': None,
        'frame': None,
        'codigo_scada_principal': None,
        'codigo_scada_alternativo': None
    }
    
    # Initialize images_processed from existing result
    images_processed = {}
    
    # Helper function to merge values (prefers non-None, concatenates if both exist)
    def merge_field_value(current, new_value):
        """Merge field values: keep non-None, concatenate if different"""
        if not new_value or str(new_value).lower() in ['none', 'null', '']:
            return current  # Keep current if new is empty
        if not current or str(current).lower() in ['none', 'null', '']:
            return new_value  # Use new if current is empty
        # Both have values - check if they're the same
        if str(current) != str(new_value):
            # Different values - concatenate
            current_str = str(current)
            new_str = str(new_value)
            if new_str not in current_str:
                return f"{current_str} + {new_str}"
        return current  # Same value, keep current
    
    # If there's imported/saved data, pre-load it
    if existing_result:
        # Extract from placa_tecnica_img* and codigo_scada_img* entries
        for key, value in existing_result.items():
            if key.startswith('placa_tecnica_img') and isinstance(value, dict):
                logger.info(f"   Loading placa data from key: {key}")
                # Extract image index from key (placa_tecnica_img0 -> 0)
                img_idx = int(key.replace('placa_tecnica_img', ''))
                # Reconstruct images_processed entry
                images_processed[img_idx] = {
                    'type': 'placa_tecnica',  # Base type
                    'method': value.get('method', 'imported'),
                    'data': value,
                    'raw_response': value.get('raw_response'),
                    'timestamp': value.get('timestamp', existing_result.get('timestamp'))
                }
                # Consolidate the data (merge with existing values)
                consolidated['marca'] = merge_field_value(consolidated['marca'], value.get('marca'))
                consolidated['modelo'] = merge_field_value(consolidated['modelo'], value.get('modelo'))
                consolidated['numero_serie'] = merge_field_value(consolidated['numero_serie'], value.get('numero_serie'))
                consolidated['año'] = merge_field_value(consolidated['año'], value.get('año'))
                consolidated['potencia'] = merge_field_value(consolidated['potencia'], value.get('potencia'))
                consolidated['voltaje'] = merge_field_value(consolidated['voltaje'], value.get('voltaje'))
                consolidated['corriente'] = merge_field_value(consolidated['corriente'], value.get('corriente'))
                consolidated['frecuencia'] = merge_field_value(consolidated['frecuencia'], value.get('frecuencia'))
                consolidated['rpm'] = merge_field_value(consolidated['rpm'], value.get('rpm'))
                consolidated['frame'] = merge_field_value(consolidated['frame'], value.get('frame'))
                
                logger.info(f"      marca: {consolidated['marca']}, modelo: {consolidated['modelo']}")
            
            elif key.startswith('codigo_scada_img') and isinstance(value, dict):
                logger.info(f"   Loading scada data from key: {key}")
                # Extract image index
                img_idx = int(key.replace('codigo_scada_img', ''))
                # Reconstruct images_processed entry
                images_processed[img_idx] = {
                    'type': 'codigo_scada',
                    'method': value.get('method', 'imported'),
                    'data': value,
                    'raw_response': value.get('raw_response'),
                    'timestamp': value.get('timestamp', existing_result.get('timestamp'))
                }
                # Consolidate the data (merge with existing values)
                consolidated['codigo_scada_principal'] = merge_field_value(
                    consolidated['codigo_scada_principal'], 
                    value.get('codigo_scada_principal')
                )
                consolidated['codigo_scada_alternativo'] = merge_field_value(
                    consolidated['codigo_scada_alternativo'],
                    value.get('codigo_scada_alternativo')
                )
                
                logger.info(f"      codigo_scada_principal: {consolidated['codigo_scada_principal']}")
    
    logger.info(f"   Final consolidated: {consolidated}")
    logger.info(f"   Restored {len(images_processed)} processed images")
    
    # Load manual edits tracking from existing result
    manual_edits = existing_result.get('manual_edits', {}) if existing_result else {}
    
    st.session_state.current_row_data = {
        'barcode': barcode,
        'images_processed': images_processed,  # Restore processed images
        'consolidated': consolidated,
        'manual_edits': manual_edits  # Track which fields were manually edited
    }
    st.session_state.current_image_index = 0
    
    logger.info(f"   ✅ Row data initialized with consolidated data and {len(images_processed)} processed images")
    logger.info(f"   ✅ Manual edits restored: {list(manual_edits.keys())}")


def consolidate_image_data(image_idx: int, image_type: str, extraction_data: dict, method: str):
    """
    Consolidate data from an image into the current row data
    Now supports multiple placas/scadas and intelligent concatenation
    
    Args:
        image_idx: Index of the image (0-based)
        image_type: 'ambos', 'placa_tecnica', 'placa_tecnica_2', 'placa_tecnica_3', 
                    'codigo_scada', 'codigo_scada_2', 'codigo_scada_3', or 'sin_datos'
        extraction_data: Data extracted from the image
        method: Extraction method used ('api', 'ocr', 'manual')
    """
    if st.session_state.current_row_data is None:
        return
    
    # Store processed image info (including raw API response if available)
    st.session_state.current_row_data['images_processed'][image_idx] = {
        'type': image_type,
        'method': method,
        'data': extraction_data,
        'raw_response': extraction_data.get('raw_response'),  # Store raw API response
        'timestamp': datetime.now().isoformat()
    }
    
    # Consolidate data based on image type
    consolidated = st.session_state.current_row_data['consolidated']
    manual_edits = st.session_state.current_row_data.get('manual_edits', {})
    
    # Helper function to concatenate values intelligently
    def concat_value(existing, new_value, separator=" + "):
        """Concatenate values if both exist, otherwise return the one that exists"""
        if not new_value or str(new_value).lower() in ['none', 'null', '']:
            return existing
        if not existing or str(existing).lower() in ['none', 'null', '']:
            return new_value
        # Both exist - concatenate if not already present
        existing_str = str(existing)
        new_str = str(new_value)
        if new_str not in existing_str:
            return f"{existing_str}{separator}{new_str}"
        return existing
    
    # Helper function to update field only if not manually edited
    def update_if_not_manual(field_name, new_value, is_secondary=False):
        """Update field only if it hasn't been manually edited"""
        if manual_edits.get(field_name):
            logger.info(f"   ⚠️ Skipping update of '{field_name}' - manually edited")
            return  # Don't update manually edited fields
        
        if is_secondary:
            # Concatenate with existing value
            consolidated[field_name] = concat_value(consolidated.get(field_name), new_value)
        else:
            # First placa or AMBOS - replace if empty, otherwise concatenate
            if not consolidated.get(field_name):
                consolidated[field_name] = new_value
            else:
                # If there's already data, concatenate
                consolidated[field_name] = concat_value(consolidated.get(field_name), new_value)
    
    # Determine if it's a placa type (including AMBOS)
    is_placa = image_type in ['ambos', 'placa_tecnica', 'placa_tecnica_2', 'placa_tecnica_3']
    is_scada = image_type in ['ambos', 'codigo_scada', 'codigo_scada_2', 'codigo_scada_3']
    
    # For secondary types (PLACA2, PLACA3), concatenate instead of replacing
    is_secondary = image_type in ['placa_tecnica_2', 'placa_tecnica_3', 'codigo_scada_2', 'codigo_scada_3']
    
    if is_placa:
        # Update placa técnica fields (respecting manual edits)
        placa_fields = ['marca', 'modelo', 'numero_serie', 'año', 'potencia', 
                        'voltaje', 'corriente', 'frecuencia', 'rpm', 'frame']
        
        for field in placa_fields:
            new_value = extraction_data.get(field)
            if new_value:
                update_if_not_manual(field, new_value, is_secondary)
    
    if is_scada:
        # Update SCADA codes (respecting manual edits)
        scada_fields = ['codigo_scada_principal', 'codigo_scada_alternativo']
        
        for field in scada_fields:
            new_value = extraction_data.get(field)
            if new_value:
                update_if_not_manual(field, new_value, is_secondary)
    
    # If 'sin_datos', don't update anything
    
    logger.info(f"✅ Consolidated data after processing image {image_idx} (type={image_type}): {consolidated}")


def get_consolidated_data() -> dict:
    """
    Get the consolidated data for the current row
    Now also checks for imported data from Procesamiento Rápido
    """
    logger.info(f"🔍 get_consolidated_data called")
    logger.info(f"   current_row_data exists: {st.session_state.current_row_data is not None}")
    
    # Priority 1: Data from current_row_data (newly processed images)
    if st.session_state.current_row_data is not None:
        consolidated = st.session_state.current_row_data['consolidated'].copy()
        logger.info(f"   Consolidated data from current_row_data: {consolidated}")
        # If we have data, return it
        if any(v for v in consolidated.values()):
            logger.info(f"   ✅ Returning data from current_row_data")
            return consolidated
    
    # Priority 2: Check if this barcode already has imported data in results
    if st.session_state.current_row_data:
        barcode = st.session_state.current_row_data.get('barcode')
        logger.info(f"   Checking for barcode: {barcode}")
        logger.info(f"   Total results: {len(st.session_state.results)}")
        
        if barcode:
            existing_result = next((r for r in st.session_state.results if r.get('BarCode') == barcode), None)
            logger.info(f"   Found existing_result: {existing_result is not None}")
            
            if existing_result:
                logger.info(f"   Existing result keys: {list(existing_result.keys())}")
                # Extract data from imported format (placa_tecnica_img0, codigo_scada_img0, etc.)
                consolidated = {}
                
                # Look for placa_tecnica_img* entries
                for key, value in existing_result.items():
                    if key.startswith('placa_tecnica_img') and isinstance(value, dict):
                        logger.info(f"   Found placa data in key: {key}, value: {value}")
                        # Merge placa data (first non-None value wins)
                        if not consolidated.get('marca'):
                            consolidated['marca'] = value.get('marca')
                        if not consolidated.get('modelo'):
                            consolidated['modelo'] = value.get('modelo')
                        if not consolidated.get('numero_serie'):
                            consolidated['numero_serie'] = value.get('numero_serie')
                        if not consolidated.get('año'):
                            consolidated['año'] = value.get('año')
                        if not consolidated.get('potencia'):
                            consolidated['potencia'] = value.get('potencia')
                        if not consolidated.get('voltaje'):
                            consolidated['voltaje'] = value.get('voltaje')
                        if not consolidated.get('corriente'):
                            consolidated['corriente'] = value.get('corriente')
                        if not consolidated.get('frecuencia'):
                            consolidated['frecuencia'] = value.get('frecuencia')
                        if not consolidated.get('rpm'):
                            consolidated['rpm'] = value.get('rpm')
                        if not consolidated.get('frame'):
                            consolidated['frame'] = value.get('frame')
                    
                    elif key.startswith('codigo_scada_img') and isinstance(value, dict):
                        logger.info(f"   Found scada data in key: {key}, value: {value}")
                        # Merge scada data
                        if not consolidated.get('codigo_scada_principal'):
                            consolidated['codigo_scada_principal'] = value.get('codigo_scada_principal')
                        if not consolidated.get('codigo_scada_alternativo'):
                            consolidated['codigo_scada_alternativo'] = value.get('codigo_scada_alternativo')
                
                logger.info(f"   Final consolidated from existing: {consolidated}")
                if any(v for v in consolidated.values()):
                    logger.info(f"   ✅ Returning data from existing_result")
                    return consolidated
    
    # Priority 3: Empty dict if no data found
    logger.info(f"   ⚠️ No data found, returning empty dict")
    return {}


def auto_save_current_row():
    """
    Auto-save current row data to results after processing
    This preserves processed data when navigating between rows
    """
    if st.session_state.current_row_data is None:
        return
    
    barcode = st.session_state.current_row_data.get('barcode')
    if not barcode:
        return
    
    # Check if we have processed images
    images_processed = st.session_state.current_row_data.get('images_processed', {})
    if not images_processed:
        return  # Nothing to save
    
    logger.info(f"💾 Auto-saving data for barcode: {barcode}")
    
    # Find existing result or create new one
    existing_idx = next((i for i, r in enumerate(st.session_state.results) if r['BarCode'] == barcode), None)
    
    # Get manual edits tracking
    manual_edits = st.session_state.current_row_data.get('manual_edits', {})
    
    # Prepare result structure
    result = {
        'BarCode': barcode,
        'metodo_extraccion': ', '.join(set(info['method'] for info in images_processed.values())),
        'auto_saved': True,  # Mark as auto-saved
        'timestamp': datetime.now().isoformat(),
        'manual_edits': manual_edits  # Persist manual edit tracking
    }
    
    # Add processed images data in the image-indexed format
    for img_idx, img_info in images_processed.items():
        image_type = img_info['type']
        extraction_data = img_info['data']
        
        # Determine the key name based on type
        if image_type == 'ambos':
            # Create both placa and scada entries
            result[f'placa_tecnica_img{img_idx}'] = {
                'tipo': 'placa_tecnica',
                'marca': extraction_data.get('marca'),
                'modelo': extraction_data.get('modelo'),
                'numero_serie': extraction_data.get('numero_serie'),
                'año': extraction_data.get('año'),
                'potencia': extraction_data.get('potencia'),
                'voltaje': extraction_data.get('voltaje'),
                'corriente': extraction_data.get('corriente'),
                'frecuencia': extraction_data.get('frecuencia'),
                'rpm': extraction_data.get('rpm'),
                'frame': extraction_data.get('frame'),
                'method': img_info['method'],
                'raw_response': img_info.get('raw_response')
            }
            result[f'codigo_scada_img{img_idx}'] = {
                'tipo': 'codigo_scada',
                'codigo_scada_principal': extraction_data.get('codigo_scada_principal'),
                'codigo_scada_alternativo': extraction_data.get('codigo_scada_alternativo'),
                'method': img_info['method'],
                'raw_response': img_info.get('raw_response')
            }
        elif image_type.startswith('placa_tecnica'):
            result[f'placa_tecnica_img{img_idx}'] = {
                'tipo': 'placa_tecnica',
                'marca': extraction_data.get('marca'),
                'modelo': extraction_data.get('modelo'),
                'numero_serie': extraction_data.get('numero_serie'),
                'año': extraction_data.get('año'),
                'potencia': extraction_data.get('potencia'),
                'voltaje': extraction_data.get('voltaje'),
                'corriente': extraction_data.get('corriente'),
                'frecuencia': extraction_data.get('frecuencia'),
                'rpm': extraction_data.get('rpm'),
                'frame': extraction_data.get('frame'),
                'method': img_info['method'],
                'raw_response': img_info.get('raw_response')
            }
        elif image_type.startswith('codigo_scada'):
            result[f'codigo_scada_img{img_idx}'] = {
                'tipo': 'codigo_scada',
                'codigo_scada_principal': extraction_data.get('codigo_scada_principal'),
                'codigo_scada_alternativo': extraction_data.get('codigo_scada_alternativo'),
                'method': img_info['method'],
                'raw_response': img_info.get('raw_response')
            }
    
    # Update or append result
    if existing_idx is not None:
        # Intelligent merge with existing data
        existing_result = st.session_state.results[existing_idx]
        
        # Keep all non-image keys from existing result
        for key, value in existing_result.items():
            if key not in result and not key.startswith(('placa_tecnica_img', 'codigo_scada_img')):
                result[key] = value
        
        # Merge image-indexed entries at field level (preserve non-None values)
        for key, value in existing_result.items():
            if key.startswith(('placa_tecnica_img', 'codigo_scada_img')):
                if key in result:
                    # Both have this image entry - merge at field level
                    existing_img_data = value
                    new_img_data = result[key]
                    
                    # For each field, keep existing value if new value is None
                    for field_key, field_value in existing_img_data.items():
                        if field_key not in new_img_data or new_img_data[field_key] is None:
                            new_img_data[field_key] = field_value
                    
                    result[key] = new_img_data
                else:
                    # Only existing has this image entry - keep it
                    result[key] = value
        
        # Update existing result
        st.session_state.results[existing_idx] = result
        logger.info(f"   ✅ Updated existing result at index {existing_idx} with intelligent merge")
    else:
        # Add new result
        st.session_state.results.append(result)
        logger.info(f"   ✅ Added new result")
    
    logger.info(f"   💾 Auto-save completed. Total results: {len(st.session_state.results)}")


def get_image_processing_status(image_idx: int) -> tuple[bool, str, str]:
    """
    Get processing status for an image
    
    Returns:
        Tuple of (processed, type, method)
    """
    if st.session_state.current_row_data is None:
        return False, '', ''
    
    images_processed = st.session_state.current_row_data['images_processed']
    if image_idx in images_processed:
        info = images_processed[image_idx]
        return True, info['type'], info['method']
    
    return False, '', ''


def save_result(row, form_data):
    """
    Save current result to session state and checkpoint (consolidates multi-image data)
    Now maintains compatibility with Procesamiento Rápido format
    Tracks manual edits to prevent overwriting user corrections
    """
    barcode = row['BarCode']
    
    # Detect manual edits by comparing form_data with consolidated data
    if st.session_state.current_row_data:
        consolidated = st.session_state.current_row_data.get('consolidated', {})
        manual_edits = st.session_state.current_row_data.get('manual_edits', {})
        
        # Check each field for manual edits
        all_fields = ['marca', 'modelo', 'numero_serie', 'año', 'potencia', 
                      'voltaje', 'corriente', 'frecuencia', 'rpm', 'frame',
                      'codigo_scada_principal', 'codigo_scada_alternativo']
        
        for field in all_fields:
            form_value = form_data.get(field)
            consolidated_value = consolidated.get(field)
            
            # Normalize values for comparison (handle None, empty strings, etc.)
            form_str = str(form_value).strip() if form_value not in [None, ''] else ''
            consol_str = str(consolidated_value).strip() if consolidated_value not in [None, ''] else ''
            
            # If values differ, mark as manually edited
            if form_str != consol_str:
                manual_edits[field] = True
                logger.info(f"   ✏️ Field '{field}' manually edited: '{consol_str}' → '{form_str}'")
        
        # Update manual_edits in current_row_data
        st.session_state.current_row_data['manual_edits'] = manual_edits
    
    # Get methods used from processed images
    methods_used = []
    if st.session_state.current_row_data and st.session_state.current_row_data['images_processed']:
        methods_used = [info['method'] for info in st.session_state.current_row_data['images_processed'].values()]
    
    # Check if this is an imported result (already has metodo_extraccion)
    existing_idx = next((i for i, r in enumerate(st.session_state.results) if r['BarCode'] == barcode), None)
    
    if existing_idx is not None:
        # Update existing result while preserving structure
        existing_result = st.session_state.results[existing_idx]
        
        # Update metodo_extraccion (preserve original if imported, otherwise use current)
        if methods_used:
            metodo_extraccion = ', '.join(set(methods_used))
        else:
            # If imported, keep original; if new, mark as manual_edit
            metodo_extraccion = existing_result.get('metodo_extraccion', 'manual_edit')
        
        # Update the result structure
        updated_result = {
            'BarCode': barcode,
            'metodo_extraccion': metodo_extraccion,
            'validated_at': datetime.now().isoformat(),
            'timestamp': existing_result.get('timestamp', datetime.now().isoformat()),
            'source': existing_result.get('source', 'manual'),
            'images_processed_count': len(st.session_state.current_row_data['images_processed']) if st.session_state.current_row_data else 0,
            'manual_edits': manual_edits  # Persist manual edit tracking
        }
        
        # Update placa_tecnica_img0 with form data (maintain format)
        if any(k.startswith('placa_tecnica_img') for k in existing_result.keys()):
            # Update existing placa entry
            placa_key = next((k for k in existing_result.keys() if k.startswith('placa_tecnica_img')), 'placa_tecnica_img0')
            placa_data = existing_result.get(placa_key, {})
            placa_data.update({
                'tipo': 'placa_tecnica',
                'marca': form_data.get('marca'),
                'modelo': form_data.get('modelo'),
                'numero_serie': form_data.get('numero_serie'),
                'año': form_data.get('año'),
                'potencia': form_data.get('potencia'),
                'voltaje': form_data.get('voltaje'),
                'corriente': form_data.get('corriente'),
                'frecuencia': form_data.get('frecuencia'),
                'rpm': form_data.get('rpm'),
                'frame': form_data.get('frame'),
                'manual_edit': True,
                'method': placa_data.get('method', 'manual_edit')
            })
            updated_result[placa_key] = placa_data
        else:
            # Create new placa entry
            updated_result['placa_tecnica_img0'] = {
                'tipo': 'placa_tecnica',
                'marca': form_data.get('marca'),
                'modelo': form_data.get('modelo'),
                'numero_serie': form_data.get('numero_serie'),
                'año': form_data.get('año'),
                'potencia': form_data.get('potencia'),
                'voltaje': form_data.get('voltaje'),
                'corriente': form_data.get('corriente'),
                'frecuencia': form_data.get('frecuencia'),
                'rpm': form_data.get('rpm'),
                'frame': form_data.get('frame'),
                'manual_edit': True,
                'method': 'manual_edit'
            }
        
        # Update codigo_scada_img0 with form data
        if any(k.startswith('codigo_scada_img') for k in existing_result.keys()):
            scada_key = next((k for k in existing_result.keys() if k.startswith('codigo_scada_img')), 'codigo_scada_img0')
            scada_data = existing_result.get(scada_key, {})
            scada_data.update({
                'tipo': 'codigo_scada',
                'codigo_scada_principal': form_data.get('codigo_scada_principal'),
                'codigo_scada_alternativo': form_data.get('codigo_scada_alternativo'),
                'manual_edit': True,
                'method': scada_data.get('method', 'manual_edit')
            })
            updated_result[scada_key] = scada_data
        else:
            # Create new scada entry if there's data
            if form_data.get('codigo_scada_principal') or form_data.get('codigo_scada_alternativo'):
                updated_result['codigo_scada_img0'] = {
                    'tipo': 'codigo_scada',
                    'codigo_scada_principal': form_data.get('codigo_scada_principal'),
                    'codigo_scada_alternativo': form_data.get('codigo_scada_alternativo'),
                    'manual_edit': True,
                    'method': 'manual_edit'
                }
        
        # Preserve other keys from existing result (like tipo_label_original, confidence, etc.)
        for key, value in existing_result.items():
            if key not in updated_result and not key.startswith(('placa_tecnica_img', 'codigo_scada_img')):
                updated_result[key] = value
        
        st.session_state.results[existing_idx] = updated_result
        
    else:
        # Create new result
        metodo_extraccion = ', '.join(set(methods_used)) if methods_used else 'manual'
        
        result = {
            'BarCode': barcode,
            'metodo_extraccion': metodo_extraccion,
            'validated_at': datetime.now().isoformat(),
            'timestamp': datetime.now().isoformat(),
            'source': 'manual',
            'images_processed_count': len(st.session_state.current_row_data['images_processed']) if st.session_state.current_row_data else 0,
            'manual_edits': manual_edits,  # Persist manual edit tracking
            'placa_tecnica_img0': {
                'tipo': 'placa_tecnica',
                'marca': form_data.get('marca'),
                'modelo': form_data.get('modelo'),
                'numero_serie': form_data.get('numero_serie'),
                'año': form_data.get('año'),
                'potencia': form_data.get('potencia'),
                'voltaje': form_data.get('voltaje'),
                'corriente': form_data.get('corriente'),
                'frecuencia': form_data.get('frecuencia'),
                'rpm': form_data.get('rpm'),
                'frame': form_data.get('frame'),
                'manual_edit': False,
                'method': metodo_extraccion
            }
        }
        
        # Add SCADA if present
        if form_data.get('codigo_scada_principal') or form_data.get('codigo_scada_alternativo'):
            result['codigo_scada_img0'] = {
                'tipo': 'codigo_scada',
                'codigo_scada_principal': form_data.get('codigo_scada_principal'),
                'codigo_scada_alternativo': form_data.get('codigo_scada_alternativo'),
                'manual_edit': False,
                'method': metodo_extraccion
            }
        
        st.session_state.results.append(result)
    
    # Auto-save checkpoint
    if hasattr(st.session_state, 'excel_path'):
        save_checkpoint(
            st.session_state.data_df,
            st.session_state.current_index,
            st.session_state.results,
            st.session_state.excel_path
        )
    
    # Reset row data for next row
    st.session_state.current_row_data = None
    st.session_state.current_image_index = 0


def display_keyboard_shortcuts():
    """Display keyboard shortcuts info"""
    st.markdown("### ⌨️ Atajos de Teclado")
    st.caption("""
    - **Enter**: Guardar y Continuar
    - **◀️ / ▶️**: Anterior / Siguiente
    - **Ctrl + S**: Guardar
    """)


# ==================== MAIN APP ====================

def main():
    initialize_session_state()
    
    st.title("📷 Transcripción Asistida Industrial v2")
    st.caption("✨ Procesamiento on-demand con checkpoints automáticos")
    
    # Check for results from other tabs (Procesamiento Rápido)
    # Allow loading checkpoints directly from Procesamiento Rápido
    st.markdown("### 📦 Importar desde Procesamiento Rápido")
    
    # Find checkpoints and batch_results from Procesamiento Rápido
    batch_results_dir = Path(st.session_state.config.output_dir) / "batch_results"
    available_imports = []
    
    if batch_results_dir.exists():
        # Find checkpoint files
        checkpoint_files = list(batch_results_dir.glob("checkpoint_*.json"))
        batch_result_files = list(batch_results_dir.glob("batch_results_*.json"))
        
        all_files = checkpoint_files + batch_result_files
        all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        available_imports = all_files[:10]  # Show last 10 files
    
    if available_imports:
        col_select, col_import = st.columns([3, 1])
        
        with col_select:
            import_options = {
                f"{f.name} ({datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})": f
                for f in available_imports
            }
            
            selected_import = st.selectbox(
                "Selecciona checkpoint o batch_results de Procesamiento Rápido:",
                options=list(import_options.keys()),
                key="select_import_file"
            )
        
        with col_import:
            if st.button("📥 Importar", use_container_width=True, key="import_batch_btn"):
                selected_file = import_options[selected_import]
                
                with st.spinner(f"Importando {selected_file.name}..."):
                    try:
                        # Load the checkpoint/batch_results file
                        with open(selected_file, 'r', encoding='utf-8') as f:
                            batch_data = json.load(f)
                        
                        # Handle both checkpoint format and batch_results format
                        if 'batch_results' in batch_data:
                            # Checkpoint format
                            batch_results = batch_data['batch_results']
                        elif isinstance(batch_data, list):
                            # Direct batch_results format
                            batch_results = batch_data
                        else:
                            st.error("❌ Formato de archivo no reconocido")
                            batch_results = []
                        
                        # Convert using the unified conversion function
                        converted_results = convert_procesamiento_to_transcripcion(batch_results)
                        
                        imported_count = 0
                        updated_count = 0
                        
                        for new_result in converted_results:
                            barcode = new_result.get('BarCode')
                            
                            # Check if barcode already exists
                            existing_idx = next((i for i, r in enumerate(st.session_state.results) if r.get('BarCode') == barcode), None)
                            
                            if existing_idx is not None:
                                # Update existing entry
                                st.session_state.results[existing_idx] = new_result
                                updated_count += 1
                                logger.info(f"Updated barcode {barcode}")
                            else:
                                # Add new entry
                                st.session_state.results.append(new_result)
                                imported_count += 1
                                logger.info(f"Imported barcode {barcode}")
                        
                        if imported_count > 0 or updated_count > 0:
                            if imported_count > 0 and updated_count > 0:
                                st.success(f"✅ Importados {imported_count} activos nuevos + Actualizados {updated_count} activos existentes")
                            elif imported_count > 0:
                                st.success(f"✅ Importados {imported_count} activos nuevos")
                            else:
                                st.success(f"✅ Actualizados {updated_count} activos existentes con datos de Procesamiento Rápido")
                            
                            # Save checkpoint with imported data (if Excel is loaded)
                            try:
                                if st.session_state.data_df is not None and st.session_state.excel_path:
                                    save_checkpoint(
                                        st.session_state.data_df,
                                        st.session_state.current_index,
                                        st.session_state.results,
                                        st.session_state.excel_path
                                    )
                                    logger.info("✅ Checkpoint saved after import")
                                else:
                                    logger.info("ℹ️ No Excel loaded, skipping checkpoint save")
                            except Exception as checkpoint_err:
                                # Don't fail the import if checkpoint save fails
                                logger.error(f"Failed to save checkpoint after import: {checkpoint_err}", exc_info=True)
                                st.warning("⚠️ Resultados importados correctamente, pero no se pudo guardar checkpoint")
                            
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.info("ℹ️ No se encontraron datos nuevos o existentes para importar")
                    
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Error al leer archivo JSON: {e}")
                    except Exception as e:
                        st.error(f"❌ Error al importar: {e}")
                        logger.error(f"Import error: {e}", exc_info=True)
    else:
        st.info("ℹ️ No hay checkpoints disponibles desde Procesamiento Rápido. Procesa algunos activos primero.")
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # API Key Configuration
        st.markdown("### 🔑 OpenAI API Key")
        
        # Check current API key
        import os
        current_key = os.getenv('OPENAI_API_KEY', '')
        
        # Show status
        if current_key:
            masked_key = f"{current_key[:10]}...{current_key[-4:]}" if len(current_key) > 14 else "***"
            st.success(f"✅ Configurada: {masked_key}")
        else:
            st.warning("⚠️ No configurada")
        
        # Expandable section for API key input
        with st.expander("🔧 Configurar API Key"):
            st.caption("Tu API key se guardará localmente en el archivo .env")
            
            new_api_key = st.text_input(
                "OpenAI API Key:",
                value="",
                type="password",
                placeholder="sk-proj-...",
                help="Obtén tu API key en: https://platform.openai.com/api-keys",
                key="api_key_input"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🧪 Probar", use_container_width=True):
                    if new_api_key.strip():
                        with st.spinner("Probando API key..."):
                            success, message = test_openai_api_key(new_api_key.strip())
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                    else:
                        st.error("⚠️ Ingresa una API key")
            
            with col2:
                if st.button("💾 Guardar", use_container_width=True):
                    if new_api_key.strip():
                        # Test first
                        with st.spinner("Validando y guardando..."):
                            success, message = test_openai_api_key(new_api_key.strip())
                            if success:
                                if save_api_key_to_env(new_api_key.strip()):
                                    st.success("✅ API Key guardada y validada")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("❌ Error al guardar")
                            else:
                                st.warning(f"⚠️ Se guardará pero: {message}")
                                if save_api_key_to_env(new_api_key.strip()):
                                    st.info("💾 Guardada (verifica que sea correcta)")
                    else:
                        st.error("⚠️ Ingresa una API key válida")
            
            with col3:
                if st.button("🔗 Obtener", use_container_width=True):
                    st.markdown(
                        '<a href="https://platform.openai.com/api-keys" target="_blank" style="text-decoration:none;">🌐 Abrir OpenAI</a>',
                        unsafe_allow_html=True
                    )
        
        st.markdown("---")
        
        # Checkpoint management
        st.markdown("### 💾 Checkpoints")
        
        checkpoints = list_available_checkpoints()
        if checkpoints:
            checkpoint_names = [
                f"{cp.stem.replace('checkpoint_', '')} ({datetime.fromtimestamp(cp.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
                for cp in checkpoints
            ]
            
            selected_checkpoint_idx = st.selectbox(
                "Checkpoints disponibles:",
                range(len(checkpoint_names)),
                format_func=lambda i: checkpoint_names[i],
                key="checkpoint_select"
            )
            
            if st.button("📂 Cargar Checkpoint", use_container_width=True):
                with st.spinner("Cargando checkpoint..."):
                    try:
                        checkpoint_data = load_checkpoint(str(checkpoints[selected_checkpoint_idx]))
                        
                        # Load Excel data
                        excel_path = checkpoint_data['excel_path']
                        st.session_state.excel_path = excel_path
                        loaded_df = load_excel_data(excel_path, st.session_state.config.output_dir)
                        
                        if loaded_df is not None:
                            st.session_state.data_df = loaded_df
                            st.session_state.current_index = checkpoint_data['current_index']
                            st.session_state.results = checkpoint_data['results']
                            st.success(f"✅ Checkpoint cargado: {len(checkpoint_data['results'])} registros completados")
                            st.rerun()
                    except ValueError as e:
                        st.error(f"⚠️ Error al cargar checkpoint: {str(e)}")
                        checkpoint_path = checkpoints[selected_checkpoint_idx]
                        if st.button("🗑️ Eliminar checkpoint corrupto", key="delete_corrupt"):
                            try:
                                checkpoint_path.unlink()
                                st.success("✅ Checkpoint corrupto eliminado")
                                time.sleep(1)
                                st.rerun()
                            except Exception as del_err:
                                st.error(f"Error al eliminar: {del_err}")
                    except Exception as e:
                        st.error(f"❌ Error inesperado: {str(e)}")
        else:
            st.info("No hay checkpoints disponibles")
        
        st.markdown("---")
        
        # File uploader
        excel_file = st.file_uploader(
            "Cargar archivo Excel",
            type=['xlsx', 'xls'],
            help="Excel con columnas 'BarCode' e imágenes"
        )
        
        if excel_file:
            # Save uploaded file
            excel_path = st.session_state.config.output_dir / excel_file.name
            with open(excel_path, 'wb') as f:
                f.write(excel_file.read())
            
            st.session_state.excel_path = str(excel_path)
            
            if st.button("🚀 Cargar Excel", use_container_width=True):
                with st.spinner("Cargando..."):
                    loaded_df = load_excel_data(str(excel_path), st.session_state.config.output_dir)
                    
                    if loaded_df is not None:
                        st.session_state.data_df = loaded_df
                        st.session_state.current_index = 0
                        st.session_state.results = []
                        st.success("✅ Excel cargado correctamente")
                        st.rerun()
        
        # Options
        if st.session_state.data_df is not None:
            st.markdown("---")
            st.markdown("### 🎯 Opciones")
            
            auto_advance = st.checkbox("Auto-avanzar al guardar", value=False, key="auto_advance")
            
            display_keyboard_shortcuts()
    
    # Main content with tabs
    # Show tabs if we have Excel loaded OR we have results (imported or processed)
    if st.session_state.data_df is not None or st.session_state.results:
        tab_transcripcion, tab_resultados = st.tabs(["📝 Transcripción", "📊 Resultados"])
        
        # === TAB 1: TRANSCRIPCIÓN ===
        with tab_transcripcion:
            process_transcription_tab()
        
        # === TAB 2: RESULTADOS ===
        with tab_resultados:
            display_results_tab()
    
    else:
        # Welcome screen
        display_welcome_screen()


def display_enhanced_image_viewer(image_paths, barcode):
    """
    Enhanced interactive image viewer with area selection and real-time OCR
    Simplified version without canvas background issues
    """
    st.markdown("#### 🖼️ Visor Interactivo con OCR")
    
    if not image_paths:
        st.info("No hay imágenes para este activo")
        return
    
    # Initialize session state
    if 'viewer_rotation' not in st.session_state:
        st.session_state.viewer_rotation = {}
    if 'viewer_scale' not in st.session_state:
        st.session_state.viewer_scale = {}
    if 'extracted_texts' not in st.session_state:
        st.session_state.extracted_texts = {}
    
    # Image selector
    selected_img_idx = st.selectbox(
        "📷 Seleccionar imagen",
        range(len(image_paths)),
        format_func=lambda x: f"Imagen {x + 1}",
        key=f"viewer_select_{barcode}"
    )
    
    if selected_img_idx >= len(image_paths):
        return
    
    img_path = image_paths[selected_img_idx]
    viewer_key = f"{barcode}_{selected_img_idx}"
    
    # Initialize controls
    if viewer_key not in st.session_state.viewer_rotation:
        st.session_state.viewer_rotation[viewer_key] = 0
    if viewer_key not in st.session_state.viewer_scale:
        st.session_state.viewer_scale[viewer_key] = 1.0
    
    # Show processing status
    is_processed, img_type, method = get_image_processing_status(selected_img_idx)
    if is_processed:
        st.success(f"✅ {method} - {img_type}")
    else:
        st.info("⏳ No procesada")
    
    # === CONTROL PANEL ===
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("↶ 90°", key=f"rotate_left_{viewer_key}", use_container_width=True):
            st.session_state.viewer_rotation[viewer_key] = (st.session_state.viewer_rotation[viewer_key] - 90) % 360
            st.rerun()
    
    with col2:
        if st.button("↻ 90°", key=f"rotate_right_{viewer_key}", use_container_width=True):
            st.session_state.viewer_rotation[viewer_key] = (st.session_state.viewer_rotation[viewer_key] + 90) % 360
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset", key=f"reset_{viewer_key}", use_container_width=True):
            st.session_state.viewer_rotation[viewer_key] = 0
            st.session_state.viewer_scale[viewer_key] = 1.0
            if viewer_key in st.session_state.extracted_texts:
                del st.session_state.extracted_texts[viewer_key]
            st.rerun()
    
    with col4:
        ocr_mode = st.selectbox(
            "OCR",
            ["manual", "full"],
            format_func=lambda x: "🎯 Área" if x == "manual" else "📄 Completo",
            key=f"ocr_mode_{viewer_key}"
        )
    
    try:
        # Load and transform image
        img = Image.open(img_path)
        
        # Apply rotation
        rotation = st.session_state.viewer_rotation[viewer_key]
        if rotation != 0:
            img = img.rotate(-rotation, expand=True)
        
        # Display image
        st.image(img, use_container_width=True, caption=f"Imagen {selected_img_idx + 1}")
        
        st.markdown("---")
        
        # === OCR MODES ===
        if ocr_mode == "manual":
            st.markdown("**🎯 Extracción por Coordenadas**")
            st.caption("Define el área rectangular para extraer texto")
            
            col_coords1, col_coords2 = st.columns(2)
            
            with col_coords1:
                x_start = st.number_input("X inicio", min_value=0, max_value=img.width, value=0, key=f"x1_{viewer_key}")
                y_start = st.number_input("Y inicio", min_value=0, max_value=img.height, value=0, key=f"y1_{viewer_key}")
            
            with col_coords2:
                x_end = st.number_input("X fin", min_value=0, max_value=img.width, value=min(300, img.width), key=f"x2_{viewer_key}")
                y_end = st.number_input("Y fin", min_value=0, max_value=img.height, value=min(200, img.height), key=f"y2_{viewer_key}")
            
            # Show preview of selected area
            if x_end > x_start and y_end > y_start:
                with st.expander("👁️ Vista previa del área"):
                    cropped_preview = img.crop((x_start, y_start, x_end, y_end))
                    st.image(cropped_preview, caption="Área seleccionada", use_container_width=True)
            
            if st.button("🔍 Extraer Texto del Área", key=f"extract_area_{viewer_key}", use_container_width=True):
                if x_end > x_start and y_end > y_start:
                    with st.spinner("Extrayendo texto..."):
                        try:
                            # Crop area
                            cropped = img.crop((x_start, y_start, x_end, y_end))
                            
                            # Initialize OCR if needed
                            if not hasattr(st.session_state, 'ocr_assistant'):
                                config = get_config()
                                st.session_state.ocr_assistant = OCRAssistant(config)
                            
                            # Extract text
                            texto_extraido = st.session_state.ocr_assistant.extract_text(cropped)
                            
                            if texto_extraido and texto_extraido.strip():
                                st.session_state.extracted_texts[viewer_key] = texto_extraido.strip()
                                st.rerun()
                            else:
                                st.warning("⚠️ No se detectó texto en el área")
                        
                        except Exception as e:
                            st.error(f"Error en OCR: {e}")
                            logger.error(f"OCR error: {e}", exc_info=True)
                else:
                    st.error("❌ Define un área válida (fin > inicio)")
        
        else:  # full mode
            st.markdown("**📄 Extracción Completa**")
            
            if st.button("🔍 Extraer Todo el Texto", key=f"extract_full_{viewer_key}", use_container_width=True):
                with st.spinner("Extrayendo texto completo..."):
                    try:
                        # Initialize OCR if needed
                        if not hasattr(st.session_state, 'ocr_assistant'):
                            config = get_config()
                            st.session_state.ocr_assistant = OCRAssistant(config)
                        
                        # Extract text
                        texto_extraido = st.session_state.ocr_assistant.extract_text(img)
                        
                        if texto_extraido and texto_extraido.strip():
                            st.session_state.extracted_texts[viewer_key] = texto_extraido.strip()
                            st.rerun()
                        else:
                            st.warning("⚠️ No se detectó texto")
                    
                    except Exception as e:
                        st.error(f"Error en OCR: {e}")
                        logger.error(f"OCR error: {e}", exc_info=True)
        
        # === DISPLAY EXTRACTED TEXT ===
        if viewer_key in st.session_state.extracted_texts:
            st.markdown("---")
            st.markdown("**📝 Texto Extraído**")
            
            texto = st.session_state.extracted_texts[viewer_key]
            
            col_text, col_clear = st.columns([5, 1])
            
            with col_text:
                st.text_area(
                    "Resultado:",
                    value=texto,
                    height=150,
                    key=f"ocr_result_{viewer_key}"
                )
            
            with col_clear:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"clear_{viewer_key}"):
                    del st.session_state.extracted_texts[viewer_key]
                    st.rerun()
            
            # Quick paste buttons
            st.markdown("**⚡ Pegar en campo:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("→ Marca", key=f"paste_marca_{viewer_key}", use_container_width=True):
                    st.session_state.quick_paste = {'field': 'marca', 'value': texto}
                    st.success("✅ Marca")
            
            with col2:
                if st.button("→ Modelo", key=f"paste_modelo_{viewer_key}", use_container_width=True):
                    st.session_state.quick_paste = {'field': 'modelo', 'value': texto}
                    st.success("✅ Modelo")
            
            with col3:
                if st.button("→ N° Serie", key=f"paste_serie_{viewer_key}", use_container_width=True):
                    st.session_state.quick_paste = {'field': 'numero_serie', 'value': texto}
                    st.success("✅ N° Serie")
            
            with col4:
                if st.button("→ SCADA", key=f"paste_scada_{viewer_key}", use_container_width=True):
                    st.session_state.quick_paste = {'field': 'codigo_scada_principal', 'value': texto}
                    st.success("✅ SCADA")
    
    except Exception as e:
        st.error(f"Error al cargar imagen: {e}")
        logger.error(f"Image viewer error: {e}", exc_info=True)


def process_transcription_tab():
    """Process the transcription tab content"""
    if st.session_state.data_df is None:
        st.info("👆 Cargue un archivo Excel o un checkpoint para comenzar la transcripción")
        return
    
    df = st.session_state.data_df
    total_rows = len(df)
    
    if total_rows == 0:
        st.error("❌ No hay filas con imágenes para procesar.")
        return
    
    current_idx = st.session_state.current_index
    
    # Safety check
    if current_idx >= total_rows:
        st.session_state.current_index = 0
        current_idx = 0
    
    # Display progress
    display_progress(current_idx, total_rows)
    
    # Get current row
    row = df.iloc[current_idx]
    image_paths = row['image_paths']
    barcode = row['BarCode']
    
    # Initialize row data if not exists
    if st.session_state.current_row_data is None or st.session_state.current_row_data['barcode'] != barcode:
        initialize_row_data(barcode)
    
    # ============================================================
    # PANEL 1: SELECCIÓN DE IMÁGENES Y MÉTODOS
    # ============================================================
    st.markdown("### 🖼️ Paso 1: Seleccionar Imágenes y Métodos de Extracción")
    st.caption(f"Activo: **{barcode}** | Total imágenes: {len(image_paths)}")
    
    # Create columns for image selection grid
    num_cols = min(3, len(image_paths))
    
    # Store selections in session state
    if 'image_selections' not in st.session_state:
        st.session_state.image_selections = {}
    
    # Initialize selections for current row
    if barcode not in st.session_state.image_selections:
        st.session_state.image_selections[barcode] = {}
    
    # Display images in grid with selection options
    for i in range(0, len(image_paths), num_cols):
        cols = st.columns(num_cols)
        
        for col_idx, img_idx in enumerate(range(i, min(i + num_cols, len(image_paths)))):
            with cols[col_idx]:
                img_path = image_paths[img_idx]
                
                # Check if already processed
                is_processed, img_type, img_method = get_image_processing_status(img_idx)
                
                # Image thumbnail
                st.markdown(f"**Imagen {img_idx + 1}**")
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
                
                # Status indicator
                if is_processed:
                    status_color = "green" if img_type != 'sin_datos' else "gray"
                    type_emoji_map = {
                        'ambos': '🔄',
                        'placa_tecnica': '📋',
                        'placa_tecnica_2': '📋📋',
                        'placa_tecnica_3': '📋📋📋',
                        'codigo_scada': '🔢',
                        'codigo_scada_2': '🔢🔢',
                        'codigo_scada_3': '🔢🔢🔢',
                        'sin_datos': '❌'
                    }
                    type_emoji = type_emoji_map.get(img_type, '❓')
                    type_display = img_type.replace('_', ' ').upper()
                    st.markdown(f":{status_color}[✅ {type_emoji} {type_display} ({img_method})]")
                else:
                    st.markdown(":orange[⏳ Pendiente]")
                
                # Selection controls
                if not is_processed:
                    # Type selector - expanded with multiple types
                    type_options = [
                        "ambos",
                        "placa_tecnica", "placa_tecnica_2", "placa_tecnica_3",
                        "codigo_scada", "codigo_scada_2", "codigo_scada_3",
                        "sin_datos"
                    ]
                    type_labels = {
                        'ambos': '🔄 AMBOS (Placa + SCADA)',
                        'placa_tecnica': '📋 PLACA 1',
                        'placa_tecnica_2': '📋 PLACA 2',
                        'placa_tecnica_3': '📋 PLACA 3',
                        'codigo_scada': '🔢 SCADA 1',
                        'codigo_scada_2': '🔢 SCADA 2',
                        'codigo_scada_3': '🔢 SCADA 3',
                        'sin_datos': '❌ Omitir'
                    }
                    img_type_select = st.selectbox(
                        "Tipo:",
                        type_options,
                        format_func=lambda x: type_labels.get(x, x),
                        key=f"type_sel_{barcode}_{img_idx}"
                    )
                    
                    # Initialize model_select to None (only used for API)
                    model_select = None
                    
                    # Method selector (if not sin_datos)
                    if img_type_select != 'sin_datos':
                        method_select = st.selectbox(
                            "Método:",
                            ["Manual", "OCR Local", "API OpenAI"],
                            key=f"method_sel_{barcode}_{img_idx}"
                        )
                        
                        # Model selector for API only
                        if method_select == "API OpenAI":
                            api_config = st.session_state.config.get('api', {}).get('assisted_mode', {})
                            available_models = api_config.get('available_models', [])
                            if available_models:
                                model_names = [m['name'] for m in available_models]
                                model_ids = [m['id'] for m in available_models]
                                sel_idx = st.selectbox(
                                    "Modelo:",
                                    range(len(model_names)),
                                    format_func=lambda i: model_names[i],
                                    key=f"api_model_{barcode}_{img_idx}"
                                )
                                model_select = model_ids[sel_idx]
                    else:
                        method_select = "Manual"
                    
                    # Store selection
                    st.session_state.image_selections[barcode][img_idx] = {
                        'type': img_type_select,
                        'method': method_select,
                        'model': model_select,
                        'path': img_path
                    }
                    
                    # Checkbox to mark for processing
                    if img_type_select != 'sin_datos' and method_select != "Manual":
                        process_checkbox = st.checkbox(
                            "✓ Procesar",
                            key=f"process_{barcode}_{img_idx}",
                            value=False
                        )
                        st.session_state.image_selections[barcode][img_idx]['process'] = process_checkbox
                else:
                    # Show reprocess button
                    if st.button("🔄 Reprocesar", key=f"reproc_{img_idx}", use_container_width=True):
                        del st.session_state.current_row_data['images_processed'][img_idx]
                        st.rerun()
    
    st.markdown("---")
    
    # Process selected images button
    images_to_process = [
        (idx, info) for idx, info in st.session_state.image_selections.get(barcode, {}).items()
        if info.get('process', False) and info['type'] != 'sin_datos' and info['method'] != 'Manual'
    ]
    
    if images_to_process:
        if st.button(f"🚀 Procesar {len(images_to_process)} Imagen(es) Seleccionada(s)", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (img_idx, info) in enumerate(images_to_process):
                status_text.text(f"Procesando imagen {img_idx + 1}...")
                
                try:
                    # Run extraction with type-specific prompt
                    # Map type to what the API expects
                    api_image_type = None  # Default to auto-detect
                    if info['type'] == 'ambos':
                        api_image_type = None  # Let API extract everything
                    elif info['type'].startswith('placa_tecnica'):
                        api_image_type = 'placa_tecnica'
                    elif info['type'].startswith('codigo_scada'):
                        api_image_type = 'codigo_scada'
                    
                    extraction_result, _ = run_extraction_on_image(
                        info['path'],
                        barcode,
                        info['method'],
                        info.get('model'),
                        api_image_type  # Pass mapped type
                    )
                    
                    # Consolidate data with the actual UI type (ambos, placa_tecnica_2, etc.)
                    method_key = info['method'].lower().replace(' ', '_').replace('api_openai', 'api')
                    consolidate_image_data(img_idx, info['type'], extraction_result, method_key)
                    
                    st.success(f"✅ Imagen {img_idx + 1} procesada")
                    
                except Exception as e:
                    st.error(f"❌ Error en imagen {img_idx + 1}: {str(e)}")
                    logger.error(f"Processing error: {e}", exc_info=True)
                
                progress_bar.progress((idx + 1) / len(images_to_process))
            
            status_text.text("✅ Procesamiento completado!")
            
            # Auto-save the processed data before rerun
            auto_save_current_row()
            
            # Trigger automatic refresh after processing
            time.sleep(1)
            st.rerun()
    
    # Mark sin_datos images automatically
    sin_datos_images = [
        idx for idx, info in st.session_state.image_selections.get(barcode, {}).items()
        if info.get('type') == 'sin_datos' and not get_image_processing_status(idx)[0]
    ]
    
    if sin_datos_images:
        if st.button(f"⏭️ Marcar {len(sin_datos_images)} Imagen(es) como Sin Datos", use_container_width=True):
            for img_idx in sin_datos_images:
                consolidate_image_data(img_idx, 'sin_datos', {}, 'manual')
            st.success(f"✅ {len(sin_datos_images)} imagen(es) marcadas como sin datos")
            st.rerun()
    
    st.markdown("---")
    
    # ============================================================
    # PANEL 2: FORMULARIO DE DATOS CON VISUALIZACIÓN
    # ============================================================
    st.markdown("### 📝 Paso 2: Revisar y Completar Datos del Activo")
    
    # Add refresh button
    col_refresh, col_spacer = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Actualizar Celdas", help="Recargar datos procesados en las celdas de edición"):
            # Rebuild consolidated data from current images_processed (not from saved results)
            if st.session_state.current_row_data and st.session_state.current_row_data.get('images_processed'):
                logger.info("🔄 Manual refresh - rebuilding consolidated from current images_processed")
                
                consolidated = {}
                images_processed = st.session_state.current_row_data['images_processed']
                manual_edits = st.session_state.current_row_data.get('manual_edits', {})
                
                # Helper function to merge field values intelligently
                def merge_value(current_val, new_val):
                    if not new_val or new_val in [None, '', 'null']:
                        return current_val
                    if not current_val or current_val in [None, '', 'null']:
                        return new_val
                    # Both have values - concatenate if different
                    current_str = str(current_val).strip()
                    new_str = str(new_val).strip()
                    if current_str == new_str:
                        return current_val
                    if new_str not in current_str:
                        return f"{current_str} + {new_str}"
                    return current_val
                
                # Rebuild from all images_processed
                for img_idx, img_info in images_processed.items():
                    extraction_data = img_info.get('data', {})
                    img_type = img_info.get('type', 'placa_tecnica')
                    
                    logger.info(f"   Processing img {img_idx}: type={img_type}, data={extraction_data}")
                    
                    # Handle placa data
                    if img_type in ['placa_tecnica', 'ambos']:
                        for field in ['marca', 'modelo', 'numero_serie', 'año', 'potencia', 
                                     'voltaje', 'corriente', 'frecuencia', 'rpm', 'frame']:
                            if field in extraction_data and extraction_data[field] not in [None, '', 'null']:
                                # Only update if not manually edited
                                if not manual_edits.get(field):
                                    consolidated[field] = merge_value(consolidated.get(field), extraction_data[field])
                    
                    # Handle scada data
                    if img_type in ['codigo_scada', 'ambos']:
                        for field in ['codigo_scada_principal', 'codigo_scada_alternativo']:
                            if field in extraction_data and extraction_data[field] not in [None, '', 'null']:
                                if not manual_edits.get(field):
                                    consolidated[field] = merge_value(consolidated.get(field), extraction_data[field])
                
                # Update the consolidated data
                st.session_state.current_row_data['consolidated'] = consolidated
                logger.info(f"✅ Consolidated rebuilt from current images_processed: {consolidated}")
            
            st.rerun()
    
    # Two columns: Form on left, images on right
    form_col, image_col = st.columns([3, 2])
    
    with form_col:
        # Use the consolidated form with API response display
        form_data = display_consolidated_form(row, barcode)
    
    with image_col:
        display_enhanced_image_viewer(image_paths, barcode)
    
    # Navigation
    display_navigation(total_rows)
    
    st.markdown("---")
    
    # === SAVE ROW BUTTON ===
    # Always show save button (removed has_processed_images condition to fix user bug)
    if st.button("💾 Guardar Fila Completa y Continuar", use_container_width=True, type="primary", key=f"save_{barcode}"):
        
        save_result(row, form_data)
        st.success("✅ Fila guardada exitosamente!")
        
        # Clear current extraction for next image
        if hasattr(st.session_state, 'current_extraction'):
            delattr(st.session_state, 'current_extraction')
        
        # Auto-advance
        if st.session_state.get('auto_advance', False) and current_idx < total_rows - 1:
            time.sleep(0.5)
            st.session_state.current_index += 1
            st.rerun()
    
    # Export results
    if st.session_state.results:
        st.sidebar.markdown("---")
        if st.sidebar.button("📥 Exportar Resultados", use_container_width=True):
            results_df = pd.DataFrame(st.session_state.results)
            output_path = st.session_state.config.output_dir / 'transcription_final_v2.csv'
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            st.sidebar.success(f"✅ Exportado a: {output_path}")


def consolidate_results_to_excel():
    """
    Consolidate results from V2 format to single-row Excel format
    Merges all placa_tecnica_img* and codigo_scada_img* entries per BarCode
    """
    if not st.session_state.results:
        st.warning("⚠️ No hay resultados para consolidar")
        return None
    
    consolidated_rows = []
    
    for result in st.session_state.results:
        barcode = result.get('BarCode')
        
        # Initialize consolidated row
        consolidated = {
            'BarCode': barcode,
            'marca': '',
            'modelo': '',
            'numero_serie': '',
            'año': '',
            'potencia': '',
            'voltaje': '',
            'corriente': '',
            'frecuencia': '',
            'rpm': '',
            'factor_potencia': '',
            'eficiencia': '',
            'ip': '',
            'clase_aislamiento': '',
            'tipo_motor': '',
            'conexion': '',
            'rodamiento_de': '',
            'rodamiento_nde': '',
            'codigo_scada_principal': '',
            'codigo_scada_respaldo': '',
            'codigo_tag': '',
            'metodo_extraccion': result.get('metodo_extraccion', 'unknown'),
            'auto_saved': result.get('auto_saved', False)
        }
        
        # Helper function to merge values with " + " separator
        def merge_value(existing, new):
            if not new or str(new).strip() == '':
                return existing
            if not existing or str(existing).strip() == '':
                return str(new).strip()
            # Check if new value is already in existing
            if str(new).strip() in str(existing):
                return existing
            return f"{existing} + {str(new).strip()}"
        
        # Merge all placa_tecnica_img* entries
        for key, value in result.items():
            if key.startswith('placa_tecnica_img') and isinstance(value, dict):
                for field in ['marca', 'modelo', 'numero_serie', 'año', 'potencia', 
                             'voltaje', 'corriente', 'frecuencia', 'rpm', 
                             'factor_potencia', 'eficiencia', 'ip', 
                             'clase_aislamiento', 'tipo_motor', 'conexion',
                             'rodamiento_de', 'rodamiento_nde']:
                    if field in value:
                        consolidated[field] = merge_value(consolidated[field], value.get(field))
        
        # Merge all codigo_scada_img* entries
        for key, value in result.items():
            if key.startswith('codigo_scada_img') and isinstance(value, dict):
                for field in ['codigo_scada_principal', 'codigo_scada_respaldo', 'codigo_tag']:
                    if field in value:
                        consolidated[field] = merge_value(consolidated[field], value.get(field))
        
        consolidated_rows.append(consolidated)
    
    # Create DataFrame
    df = pd.DataFrame(consolidated_rows)
    
    # Reorder columns for better readability
    column_order = [
        'BarCode',
        'marca', 'modelo', 'numero_serie', 'año',
        'potencia', 'voltaje', 'corriente', 'frecuencia', 'rpm',
        'factor_potencia', 'eficiencia', 'ip', 'clase_aislamiento',
        'tipo_motor', 'conexion',
        'rodamiento_de', 'rodamiento_nde',
        'codigo_scada_principal', 'codigo_scada_respaldo', 'codigo_tag',
        'metodo_extraccion', 'auto_saved'
    ]
    
    df = df[column_order]
    
    return df


def display_results_tab():
    """Display results tab - similar to Procesamiento Rápido"""
    if not st.session_state.results:
        st.info("ℹ️ No hay resultados aún. Procesa algunos activos en la pestaña 'Transcripción' o importa datos desde Procesamiento Rápido.")
        return
    
    st.markdown("### 📊 Resultados de la Transcripción Asistida")
    
    # Summary metrics
    total_activos = len(st.session_state.results)
    
    # Count how many have any data (not just imported)
    activos_con_datos = sum(1 for r in st.session_state.results if any(
        k.startswith('placa_tecnica_img') or k.startswith('codigo_scada_img') 
        for k in r.keys()
    ))
    
    # Count methods used
    methods_used = {}
    for result in st.session_state.results:
        method = result.get('metodo_extraccion', 'unknown')
        methods_used[method] = methods_used.get(method, 0) + 1
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Activos", total_activos)
    with col2:
        st.metric("Con Datos", activos_con_datos)
    with col3:
        if methods_used:
            top_method = max(methods_used.items(), key=lambda x: x[1])
            st.metric("Método Principal", f"{top_method[0][:15]}..." if len(top_method[0]) > 15 else top_method[0], f"{top_method[1]} activos")
    
    st.markdown("---")
    
    # Quick search
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_barcode = st.text_input("🔍 Buscar por BarCode", key="search_transcription_barcode", placeholder="Ingresa BarCode para filtrar...")
    with search_col2:
        if st.button("🔄 Ver Todos", key="clear_search_transcription"):
            st.session_state['search_transcription_barcode'] = ""
            st.rerun()
    
    # Results table - convert to PLACA/SCADA format for display
    results_data = []
    for result in st.session_state.results:
        barcode = result.get('BarCode')
        
        # Check for placa_tecnica entries
        placa_entries = {k: v for k, v in result.items() if k.startswith('placa_tecnica_img')}
        scada_entries = {k: v for k, v in result.items() if k.startswith('codigo_scada_img')}
        
        if placa_entries:
            for key, data in placa_entries.items():
                img_idx = key.replace('placa_tecnica_img', '')
                tipo_label = f"PLACA{int(img_idx) + 1}" if img_idx.isdigit() else "PLACA"
                
                results_data.append({
                    'BarCode': barcode,
                    'Tipo': tipo_label,
                    'Marca': data.get('marca', ''),
                    'Modelo': data.get('modelo', ''),
                    'Potencia': data.get('potencia', ''),
                    'Número de Serie': data.get('numero_serie', ''),
                    'Año': data.get('año', ''),
                    'Código SCADA': '',
                    'Método': data.get('method', result.get('metodo_extraccion', 'unknown'))[:20],
                    'Editado': '✏️' if data.get('manual_edit') else ''
                })
        
        if scada_entries:
            for key, data in scada_entries.items():
                img_idx = key.replace('codigo_scada_img', '')
                tipo_label = f"SCADA{int(img_idx) + 1}" if img_idx.isdigit() else "SCADA"
                
                results_data.append({
                    'BarCode': barcode,
                    'Tipo': tipo_label,
                    'Marca': '',
                    'Modelo': '',
                    'Potencia': '',
                    'Número de Serie': '',
                    'Año': '',
                    'Código SCADA': data.get('codigo_scada_principal', ''),
                    'Método': data.get('method', result.get('metodo_extraccion', 'unknown'))[:20],
                    'Editado': '✏️' if data.get('manual_edit') else ''
                })
        
        # If no placa or scada entries (shouldn't happen), show basic info
        if not placa_entries and not scada_entries:
            results_data.append({
                'BarCode': barcode,
                'Tipo': 'N/A',
                'Marca': '',
                'Modelo': '',
                'Potencia': '',
                'Número de Serie': '',
                'Año': '',
                'Código SCADA': '',
                'Método': result.get('metodo_extraccion', 'unknown')[:20],
                'Editado': ''
            })
    
    df_results = pd.DataFrame(results_data)
    
    # Apply search filter if specified
    if search_barcode and search_barcode.strip():
        df_results = df_results[df_results['BarCode'].str.contains(search_barcode.strip(), case=False, na=False)]
        st.caption(f"📊 Mostrando {len(df_results)} resultados que coinciden con '{search_barcode.strip()}'")
    
    st.dataframe(df_results, use_container_width=True, height=600)
    
    # Export options
    st.markdown("---")
    st.markdown("### 💾 Opciones de Exportación")
    
    col_export1, col_export2, col_export3, col_export4 = st.columns([1, 1, 1, 1])
    
    with col_export1:
        if st.button("📊 Exportar Excel Consolidado", key="export_excel_consolidated", type="primary", use_container_width=True):
            try:
                # Check if openpyxl is available
                try:
                    import openpyxl
                except ImportError:
                    st.error("❌ Openpyxl no está instalado. Instálalo con: pip install openpyxl")
                    return
                
                df_consolidated = consolidate_results_to_excel()
                if df_consolidated is not None:
                    output_path = st.session_state.config.output_dir / 'transcription_consolidated.xlsx'
                    df_consolidated.to_excel(output_path, index=False, engine='openpyxl')
                    st.success(f"✅ Excel consolidado exportado a: {output_path.name}")
                    st.info(f"📊 {len(df_consolidated)} activos consolidados (una fila por BarCode)")
            except Exception as e:
                st.error(f"❌ Error al exportar Excel: {e}")
                logger.error(f"Export consolidated error: {e}", exc_info=True)
    
    with col_export2:
        if st.button("💾 Exportar CSV", key="export_csv_transcription", type="secondary", use_container_width=True):
            output_path = st.session_state.config.output_dir / 'transcription_results.csv'
            df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
            st.success(f"✅ CSV exportado a: {output_path.name}")
    
    with col_export3:
        if st.button("📋 Exportar JSON", key="export_json_transcription", type="secondary", use_container_width=True):
            output_path = st.session_state.config.output_dir / 'transcription_results.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.results, f, ensure_ascii=False, indent=2, default=str)
            st.success(f"✅ JSON exportado a: {output_path.name}")
    
    with col_export4:
        if st.button("💾 Guardar Checkpoint", key="save_checkpoint_manual", type="secondary", use_container_width=True):
            if st.session_state.data_df is not None and st.session_state.excel_path:
                try:
                    save_checkpoint(
                        st.session_state.data_df,
                        st.session_state.current_index,
                        st.session_state.results,
                        st.session_state.excel_path
                    )
                    st.success("✅ Checkpoint guardado correctamente")
                except Exception as e:
                    st.error(f"❌ Error al guardar checkpoint: {e}")
            else:
                st.warning("⚠️ No hay Excel cargado para guardar checkpoint")
    
    # Method statistics
    if methods_used:
        st.markdown("---")
        st.markdown("### 📈 Estadísticas por Método")
        method_cols = st.columns(min(len(methods_used), 4))
        for idx, (method, count) in enumerate(methods_used.items()):
            if idx < 4:  # Limit to 4 columns
                with method_cols[idx]:
                    emoji = "🤖" if "api" in method.lower() else "🔍" if "ocr" in method.lower() else "📥" if "import" in method.lower() else "✏️"
                    method_display = method.replace('imported_from_procesamiento_rapido', 'Importado')[:25]
                    st.metric(f"{emoji} {method_display}", count)


def display_welcome_screen():
    """Display welcome screen when no data is loaded"""
    st.info("👆 Cargue un archivo Excel o un checkpoint para comenzar")
    
    st.markdown("""
    ### ✨ Novedades v2:
    - ✅ **Procesamiento on-demand**: No preprocesa todo al inicio
    - ✅ **Método por imagen**: Elige OCR o API para cada imagen
    - ✅ **Checkpoints automáticos**: Guarda progreso después de cada imagen
    - ✅ **Continuar desde checkpoint**: Retoma tu trabajo donde lo dejaste
    - ✅ **Más rápido**: Comienza a trabajar inmediatamente
    - ✅ **Más flexible**: Cambia de método según necesites
    - ✅ **Tab de Resultados**: Visualiza todos tus resultados procesados o importados
    
    ### 🎯 Características:
    - ✅ OCR automático con EasyOCR
    - ✅ API OpenAI (GPT-4o Mini, GPT-4o, GPT-4 Turbo)
    - ✅ Pre-llenado inteligente de formularios
    - ✅ Código de colores por confianza
    - ✅ Visualización original y preprocesada
    - ✅ Auto-guardado continuo con checkpoints
    - ✅ Atajos de teclado
    - ✅ Estadísticas en tiempo real
    - ✅ Compatible 100% con Procesamiento Rápido
    
    ### 📖 Cómo usar:
    1. Cargue su archivo Excel o continúe desde un checkpoint
    2. Para cada imagen:
       - Seleccione el método (OCR Local o API OpenAI)
       - Presione "Extraer"
       - Revise/corrija los campos pre-llenados
       - Guarde y continúe
    3. El progreso se guarda automáticamente
    4. Exporte los resultados finales cuando termine
    5. Use el tab "📊 Resultados" para ver y exportar todos los datos
    
    ### 🎨 Código de Colores:
    - 🟢 **Verde**: Alta confianza (>85%) - Probablemente correcto
    - 🟡 **Amarillo**: Media confianza (60-85%) - Revisar
    - 🔴 **Rojo**: Baja confianza (<60%) - Requiere corrección
    """)


if __name__ == "__main__":
    main()
