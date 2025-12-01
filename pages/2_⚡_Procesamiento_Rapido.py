"""
Procesamiento Rápido en Batch
Permite procesar múltiples imágenes de múltiples activos de forma rápida
"""

import streamlit as st
import pandas as pd
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from PIL import Image
import logging
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared_results import SharedResultsManager

from excel_image_extractor import ExcelImageExtractor
from api_extractor import APIExtractor
from image_preprocessor import ImagePreprocessor
from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Procesamiento Rápido",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Disable the dimming/transparency effect when buttons are clicked */
    .stApp [data-testid="stAppViewContainer"] {
        opacity: 1 !important;
    }
    
    .stApp [data-testid="stAppViewContainer"] > section {
        opacity: 1 !important;
    }
    
    /* Keep content fully visible during interactions */
    .element-container {
        opacity: 1 !important;
    }
    
    /* Prevent dimming during form submission or button clicks */
    .stApp.stAppProcessing [data-testid="stAppViewContainer"] {
        opacity: 1 !important;
    }
    
    .small-img {
        max-height: 80px;
        object-fit: contain;
    }
    .row-container {
        border: 1px solid #ddd;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .processed {
        background-color: #d4edda;
    }
    .pending {
        background-color: #fff3cd;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state"""
    logger.info("🔧 initialize_session_state() called")
    
    if 'batch_initialized' not in st.session_state:
        logger.info("📦 First initialization - creating new session state")
        st.session_state.batch_initialized = True
        st.session_state.batch_data_df = None
        st.session_state.batch_results = []
        st.session_state.batch_config = get_config()
        st.session_state.batch_selections = {}  # {barcode: {img_idx: {'process': bool, 'type': str, 'method': str}}}
        st.session_state.batch_processing_status = {}  # {barcode: 'pending'|'processing'|'completed'|'error'}
        st.session_state.checkpoint_restored = None
        
        # Initialize queue variables (critical for incremental processing)
        st.session_state.processing_queue = []
        st.session_state.individual_results = {}
        
        logger.info(f"✅ Initialized: processing_queue={st.session_state.processing_queue}, individual_results={st.session_state.individual_results}")
        
        # Try to restore checkpoint (will override above if checkpoint exists)
        restore_checkpoint_incremental()
    else:
        logger.info(f"♻️ Session already initialized. Queue length: {len(st.session_state.get('processing_queue', []))}")


def load_excel_data_batch(excel_path: str, output_dir: Path):
    """Load and extract images from Excel for batch processing"""
    with st.spinner("Cargando datos del Excel..."):
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
            st.error("No se encontraron imágenes en el Excel")
            return None
        
        st.success(f"✅ {len(df_with_images)} activos con {stats['total_images']} imágenes")
        return df_with_images


def prepare_selector_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert image data to editable DataFrame format
    Flattens all images into individual rows for st.data_editor
    """
    rows = []
    
    # CRITICAL: Use GLOBAL img_idx counter to prevent row_key collisions across pages
    global_img_idx = 0
    
    for idx, row in df.iterrows():
        barcode = row['BarCode']
        image_paths = row['image_paths']
        
        for local_idx, img_path in enumerate(image_paths):
            # Convert to string path (Streamlit needs string paths)
            path_str = str(Path(img_path).resolve())  # Absolute path
            
            rows.append({
                '🖼️': path_str,  # Image column for thumbnail
                'BarCode': barcode,
                'Img': f'{local_idx + 1}',
                '✅': False,  # Checkbox (shorter header)
                'Tipo': 'PLACA1',  # Default value - user can edit to PLACA2, SCADA1, etc.
                '_img_idx': global_img_idx,  # GLOBAL index across all images
                '_local_img_idx': local_idx,  # Local index within this barcode (for display)
                '_img_path': path_str,
                '_barcode_orig': barcode
            })
            
            global_img_idx += 1  # Increment GLOBAL counter
    
    return pd.DataFrame(rows)


def display_batch_row_selector(df: pd.DataFrame):
    """Display ultra-fast batch selector using st.data_editor (no re-renders!)"""
    st.subheader("📋 Configuración de Procesamiento Batch")
    st.caption("⚡ Tabla editable ultra-rápida - maneja miles de imágenes sin esperas. Usa checkboxes y dropdowns directamente.")
    
    # ==== CONFIGURACIÓN GLOBAL ====
    st.markdown("### ⚙️ Configuración Global")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Method selector (stored in session state to persist without rerun)
        if 'batch_global_method' not in st.session_state:
            st.session_state.batch_global_method = "API OpenAI"
        
        default_method = st.selectbox(
            "Método de procesamiento",
            ["API OpenAI", "OCR Local", "Manual"],
            index=["API OpenAI", "OCR Local", "Manual"].index(st.session_state.batch_global_method),
            key="batch_default_method_selector",
            help="Método que se usará para todas las imágenes seleccionadas"
        )
        st.session_state.batch_global_method = default_method
    
    with col2:
        if default_method == "API OpenAI":
            # Get available models from config
            available_models = st.session_state.batch_config.get('api', {}).get('assisted_mode', {}).get('available_models', [])
            
            if 'batch_global_model' not in st.session_state:
                # Find recommended model as default
                for model in available_models:
                    if model.get('recommended', False):
                        st.session_state.batch_global_model = model['id']
                        break
            
            if available_models:
                # Create model options with cost info
                model_options = []
                model_ids = {}
                
                for model in available_models:
                    name = model['name']
                    cost = model.get('cost_per_1k_tokens', 0) * 1000
                    recommended = model.get('recommended', False)
                    
                    label = f"{name}"
                    if recommended:
                        label += " ⭐"
                    label += f" (${cost:.2f}/1k imgs)"
                    
                    model_options.append(label)
                    model_ids[label] = model['id']
                
                # Find current selection index
                current_model = st.session_state.get('batch_global_model', available_models[0]['id'])
                try:
                    current_idx = [model_ids[opt] for opt in model_options].index(current_model)
                except ValueError:
                    current_idx = 0
                
                selected_label = st.selectbox(
                    "Modelo API",
                    model_options,
                    index=current_idx,
                    key="batch_default_model_selector",
                    help="⭐ = Recomendado. Costo aproximado por 1000 imágenes."
                )
                
                default_model = model_ids.get(selected_label, available_models[0]['id'])
                st.session_state.batch_global_model = default_model
                
                # Show model description
                selected_model = next((m for m in available_models if m['id'] == default_model), None)
                if selected_model:
                    st.caption(f"ℹ️ {selected_model.get('description', '')}")
            else:
                # Fallback
                default_model = st.selectbox(
                    "Modelo API",
                    ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "gpt-4-turbo-2024-04-09"],
                    key="batch_default_model_fallback"
                )
                st.session_state.batch_global_model = default_model
        else:
            default_model = None
    
    st.markdown("---")
    
    # ==== PREPARE DATA EDITOR ====
    # Initialize or load selector DataFrame
    if 'batch_selector_df' not in st.session_state or st.session_state.get('batch_df_needs_refresh', False):
        st.session_state.batch_selector_df = prepare_selector_dataframe(df)
        st.session_state.batch_df_needs_refresh = False
    
    selector_df = st.session_state.batch_selector_df.copy()
    
    # ==== SELECCIÓN MASIVA ====
    st.markdown("### 🎯 Acciones Rápidas")
    st.caption("Usa estos botones para configurar todas las imágenes de una vez")
    
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    
    with col_a:
        if st.button("✅ Todas → PLACA", use_container_width=True, help="Marcar todas las imágenes como Placa Técnica"):
            st.session_state.batch_selector_df['✅'] = True
            st.session_state.batch_selector_df['Tipo'] = 'PLACA1'
            save_configuration(st.session_state.batch_selector_df)
            st.success("✅ Todas marcadas como PLACA1", icon="✅")
            st.rerun()
    
    with col_b:
        if st.button("🔢 Todas → SCADA", use_container_width=True, help="Marcar todas las imágenes como Código SCADA"):
            st.session_state.batch_selector_df['✅'] = True
            st.session_state.batch_selector_df['Tipo'] = 'SCADA1'
            save_configuration(st.session_state.batch_selector_df)
            st.success("🔢 Todas marcadas como SCADA1", icon="🔢")
            st.rerun()
    
    with col_c:
        if st.button("🔄 Todas → AMBOS", use_container_width=True, help="Marcar todas las imágenes para extraer PLACA + SCADA"):
            st.session_state.batch_selector_df['✅'] = True
            st.session_state.batch_selector_df['Tipo'] = 'AMBOS'
            save_configuration(st.session_state.batch_selector_df)
            st.success("🔄 Todas marcadas como AMBOS", icon="🔄")
            st.rerun()
    
    with col_d:
        if st.button("🚫 Deseleccionar", use_container_width=True, help="Quitar todas las marcas de procesamiento"):
            st.session_state.batch_selector_df['✅'] = False
            save_configuration(st.session_state.batch_selector_df)
            st.info("🚫 Todas deseleccionadas", icon="🚫")
            st.rerun()
    
    with col_e:
        if st.button("🗑️ Reset", use_container_width=True, help="Resetear toda la configuración"):
            st.session_state.batch_df_needs_refresh = True
            if 'batch_selector_df' in st.session_state:
                save_configuration(st.session_state.batch_selector_df)
            st.warning("🗑️ Configuración reseteada", icon="🗑️")
            st.rerun()
    
    st.markdown("---")
    
    # ==== SELECTOR DE RANGO DE FILAS ====
    st.markdown("### 📏 Selección por Rango (para lotes)")
    st.caption("💡 **Recomendado para datasets grandes**: Procesa en lotes de 300-500 imágenes para evitar timeouts")
    
    range_col1, range_col2, range_col3, range_col4 = st.columns([2, 2, 2, 2])
    
    with range_col1:
        start_row = st.number_input(
            "Desde fila",
            min_value=1,
            max_value=len(selector_df),
            value=1,
            step=1,
            help="Fila inicial del rango (inclusive)"
        )
    
    with range_col2:
        end_row = st.number_input(
            "Hasta fila",
            min_value=1,
            max_value=len(selector_df),
            value=min(500, len(selector_df)),
            step=1,
            help="Fila final del rango (inclusive)"
        )
    
    with range_col3:
        range_tipo = st.selectbox(
            "Tipo para rango",
            options=['PLACA1', 'PLACA2', 'SCADA1', 'SCADA2', 'AMBOS'],
            help="Tipo que se asignará a todas las filas del rango"
        )
    
    with range_col4:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("✅ Marcar Rango", use_container_width=True, type="primary"):
            if start_row > end_row:
                st.error("❌ La fila inicial debe ser menor o igual que la fila final")
            else:
                # Convert to 0-indexed
                idx_start = start_row - 1
                idx_end = end_row
                
                # Mark the range
                st.session_state.batch_selector_df.iloc[idx_start:idx_end, st.session_state.batch_selector_df.columns.get_loc('✅')] = True
                st.session_state.batch_selector_df.iloc[idx_start:idx_end, st.session_state.batch_selector_df.columns.get_loc('Tipo')] = range_tipo
                
                save_configuration(st.session_state.batch_selector_df)
                num_marked = end_row - start_row + 1
                st.success(f"✅ Marcadas {num_marked} filas ({start_row} a {end_row}) como {range_tipo}", icon="✅")
                st.rerun()
    
    st.markdown("---")
    
    # ==== TABLA INTEGRADA: CONFIGURAR + PROCESAR + REVISAR ====
    st.markdown("### 📊 Procesamiento y Revisión Integrada")
    st.caption("✨ **Workflow completo**: Configura → Procesa → Revisa → Corrige → Guarda (todo en la misma tabla)")
    
    # === CHECKPOINT / PROGRESS RESTORE SECTION ===
    st.markdown("---")
    
    checkpoint_col1, checkpoint_col2, checkpoint_col3 = st.columns([2, 2, 1])
    
    with checkpoint_col1:
        st.markdown("#### 📂 Cargar Progreso Anterior")
        
        # Allow user to select checkpoint directory
        checkpoint_dir_col1, checkpoint_dir_col2 = st.columns([4, 1])
        
        with checkpoint_dir_col1:
            # Default directory
            default_checkpoint_dir = str(Path(st.session_state.batch_config.output_dir) / "batch_results") if hasattr(st.session_state, 'batch_config') else ""
            
            checkpoint_directory = st.text_input(
                "Directorio de checkpoints:",
                value=st.session_state.get('checkpoint_load_directory', default_checkpoint_dir),
                key="checkpoint_load_dir_input",
                help="Ruta donde se encuentran los archivos de checkpoint"
            )
            st.session_state.checkpoint_load_directory = checkpoint_directory
        
        with checkpoint_dir_col2:
            if st.button("📁 Examinar", key="browse_checkpoint_dir", help="Seleccionar directorio"):
                st.info("💡 Copia y pega la ruta del directorio en el campo de texto")
        
        # Check for available checkpoints in selected directory
        checkpoint_files = []
        if checkpoint_directory and Path(checkpoint_directory).exists():
            output_dir = Path(checkpoint_directory)
            
            # Find ALL valid checkpoint/result files
            valid_patterns = [
                "checkpoint_incremental_latest.json",     # Latest incremental checkpoint
                "checkpoint_*.json",                      # Backup checkpoints
                "batch_results_*.json",                   # Batch processing results
                "corrected_results_*.json"                # Corrected results
            ]
            
            found_files = set()
            
            # Latest checkpoint (highest priority)
            latest = output_dir / "checkpoint_incremental_latest.json"
            if latest.exists():
                found_files.add(latest)
            
            # Search for all patterns
            for pattern in valid_patterns[1:]:  # Skip the latest which we already checked
                for f in output_dir.glob(pattern):
                    if f.name != "checkpoint_incremental_latest.json":  # Avoid duplicates
                        found_files.add(f)
            
            # Sort by modification time (newest first)
            checkpoint_files = sorted(found_files, key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Limit to most recent 15 files
            checkpoint_files = checkpoint_files[:15]
        
        if checkpoint_files:
            file_options = {}
            for f in checkpoint_files:
                # Show file info with enhanced detection
                try:
                    file_size = f.stat().st_size / 1024  # KB
                    file_time = datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                    
                    # Try to read and analyze file content
                    try:
                        # Try UTF-8 first, then fallback with error handling
                        try:
                            with open(f, 'r', encoding='utf-8') as fp:
                                data = json.load(fp)
                        except UnicodeDecodeError:
                            # Fallback to UTF-8 with error handling
                            with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
                                data = json.load(fp)
                            logger.warning(f"Loaded {f.name} with encoding errors ignored")
                        
                        # Determine file type and content
                        if isinstance(data, dict) and 'batch_results' in data:
                            # Checkpoint format
                            meta = data.get('metadata', {})
                            batch_count = len(data.get('batch_results', []))
                            individual_count = len(data.get('individual_results', {}))
                            
                            file_type = "📦 Checkpoint"
                            if meta:
                                label = f"{file_type} - {f.name} ({file_time}) - {batch_count} activos, {individual_count} procesados"
                            else:
                                label = f"{file_type} - {f.name} ({file_time}) - {batch_count} activos"
                                
                        elif isinstance(data, list):
                            # Array of results (batch_results format)
                            result_count = len(data)
                            file_type = "📊 Resultados"
                            
                            # Count images processed
                            total_images = 0
                            for result in data:
                                if 'results_by_type' in result:
                                    total_images += len(result['results_by_type'])
                                elif 'images_processed' in result:
                                    total_images += len(result['images_processed'])
                            
                            label = f"{file_type} - {f.name} ({file_time}) - {result_count} activos, {total_images} imágenes"
                            
                        elif isinstance(data, dict) and len(data) == 1:
                            # Wrapped format
                            wrapped_data = list(data.values())[0]
                            if isinstance(wrapped_data, list):
                                result_count = len(wrapped_data)
                                file_type = "📋 Datos"
                                label = f"{file_type} - {f.name} ({file_time}) - {result_count} registros"
                            else:
                                file_type = "📄 Archivo"
                                label = f"{file_type} - {f.name} ({file_time}) - {file_size:.1f}KB"
                        else:
                            # Unknown format
                            file_type = "❓ Otro"
                            label = f"{file_type} - {f.name} ({file_time}) - {file_size:.1f}KB"
                            
                    except Exception as parse_error:
                        # Can't parse file - show specific error
                        file_type = "⚠️ Error"
                        error_msg = str(parse_error)[:50]  # First 50 chars of error
                        label = f"{file_type} - {f.name} ({file_time}) - {file_size:.1f}KB ({error_msg}...)"
                        logger.warning(f"Could not parse {f.name}: {parse_error}")
                    
                    file_options[label] = f
                    
                except Exception as e:
                    # File system error
                    file_options[f"❌ {f.name} (error de acceso)"] = f
                    logger.error(f"Could not access {f.name}: {e}")
            
            selected_checkpoint = st.selectbox(
                "Selecciona un archivo de checkpoint o resultados:",
                options=list(file_options.keys()),
                key="checkpoint_selector",
                help="Compatible con: checkpoint_*.json, batch_results_*.json, corrected_results_*.json"
            )
            
            load_col1, load_col2 = st.columns([2, 1])
            with load_col1:
                if st.button("🔄 Cargar y Continuar", use_container_width=True, type="primary", key="load_checkpoint_btn"):
                    selected_file = file_options[selected_checkpoint]
                    if load_results_from_file(selected_file):
                        st.info("💡 Datos cargados - Verifica el progreso en todas las pestañas")
                        time.sleep(1)
                        st.rerun()
            
            with load_col2:
                if st.button("🗑️ Limpiar", use_container_width=True, key="clear_checkpoint_btn", help="Borrar estado actual y empezar de cero"):
                    clear_checkpoint_incremental()
                    st.rerun()
        else:
            st.info("📁 No se encontraron archivos compatibles en el directorio seleccionado.")
            st.caption("📝 **Formatos compatibles:** checkpoint_*.json, batch_results_*.json, corrected_results_*.json")
            if checkpoint_directory:
                st.caption(f"🔍 **Buscando en:** {checkpoint_directory}")
                # Show what files exist in the directory
                try:
                    existing_files = list(Path(checkpoint_directory).glob("*.json"))
                    if existing_files:
                        st.caption(f"📋 **Archivos JSON encontrados:** {', '.join([f.name for f in existing_files[:5]])}")
                        if len(existing_files) > 5:
                            st.caption(f"... y {len(existing_files) - 5} más")
                    else:
                        st.caption("❌ **Sin archivos JSON** en el directorio")
                except Exception as e:
                    st.caption(f"⚠️ Error al listar archivos: {e}")
    
    with checkpoint_col2:
        st.markdown("#### 📊 Estado Actual")
        if st.session_state.batch_results:
            st.metric("Activos Procesados", len(st.session_state.batch_results))
            total_images = sum(len(r.get('images_processed', [])) for r in st.session_state.batch_results)
            st.metric("Imágenes Totales", total_images)
        else:
            st.info("Sin progreso aún")
    
    with checkpoint_col3:
        st.markdown("#### 💾 Guardar")
        
        # Allow user to select save directory
        save_dir_default = str(Path(st.session_state.batch_config.output_dir) / "batch_results") if hasattr(st.session_state, 'batch_config') else ""
        
        checkpoint_save_directory = st.text_input(
            "Directorio:",
            value=st.session_state.get('checkpoint_save_directory', save_dir_default),
            key="checkpoint_save_dir_input",
            help="Directorio donde se guardarán los checkpoints"
        )
        st.session_state.checkpoint_save_directory = checkpoint_save_directory
        
        if st.button("💾 Guardar Ahora", use_container_width=True, key="manual_checkpoint_save_new"):
            save_checkpoint_incremental()
            st.success("✅ Guardado")
            time.sleep(0.5)
            st.rerun()
    
    st.markdown("---")
    
    # Check for results from other tabs (Transcripción Asistida)
    if hasattr(st.session_state, 'batch_config'):
        try:
            shared_manager = SharedResultsManager(st.session_state.batch_config.output_dir)
            stats = shared_manager.get_statistics()
            
            if stats['by_source'].get('transcripcion_asistida', 0) > 0 or stats['by_source'].get('both', 0) > 0:
                col_info1, col_info2 = st.columns([3, 1])
                with col_info1:
                    st.info(f"🔄 **Resultados compartidos detectados**: {stats['by_source'].get('transcripcion_asistida', 0)} desde Transcripción Asistida, {stats['by_source'].get('both', 0)} desde ambos tabs")
                with col_info2:
                    if st.button("📥 Cargar Resultados Compartidos", use_container_width=True):
                        # Import shared results
                        shared_batch_results = shared_manager.export_for_procesamiento_rapido()
                        
                        # Merge with existing results
                        if 'batch_results' not in st.session_state:
                            st.session_state.batch_results = []
                        
                        # Deduplicate by BarCode
                        existing_barcodes = {r['BarCode'] for r in st.session_state.batch_results}
                        new_results = [r for r in shared_batch_results if r['BarCode'] not in existing_barcodes]
                        
                        st.session_state.batch_results.extend(new_results)
                        st.success(f"✅ Cargados {len(new_results)} resultados nuevos desde archivo compartido")
                        st.rerun()
        except Exception as e:
            logger.warning(f"Could not check shared results: {e}")
    
    # Warning if there are pending edits
    if st.session_state.get('pending_edits'):
        st.warning(f"⚠️ **{len(st.session_state.pending_edits)} edición(es) pendiente(s)** - No olvides guardar antes de cambiar de página. Se auto-guardarán al navegar.")
    
    # Show progress summary if there are results
    if 'individual_results' in st.session_state and st.session_state.individual_results:
        successful_results = {k: v for k, v in st.session_state.individual_results.items() if v.get('status') == 'success'}
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        with col_sum1:
            st.metric("✅ Procesadas", len(successful_results))
        with col_sum2:
            st.metric("⚪ Pendientes", len(selector_df) - len(successful_results))
        with col_sum3:
            en_cola = len(st.session_state.get('processing_queue', []))
            st.metric("⏳ En Procesamiento", en_cola)
        with col_sum4:
            if st.button("💾 Exportar a Excel", use_container_width=True, type="primary"):
                export_results()
    
    # ==== TABLA VISUAL CON MINIATURAS (SIN RE-RENDERS) ====
    
    # Initialize pagination state
    if 'batch_page_number' not in st.session_state:
        st.session_state.batch_page_number = 1
    if 'batch_page_size' not in st.session_state:
        st.session_state.batch_page_size = 100
    if 'batch_pending_changes' not in st.session_state:
        st.session_state.batch_pending_changes = {}  # Store pending changes before form submit
    if 'individual_results' not in st.session_state:
        st.session_state.individual_results = {}  # Store results for each processed image
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []  # Consolidated results for export
    if 'processing_queue' not in st.session_state:
        st.session_state.processing_queue = []  # Queue of images being processed
    
    if 'pending_edits' not in st.session_state:
        st.session_state.pending_edits = set()  # Track rows with unsaved edits
    
    # Display in compact rows with images
    if len(selector_df) > 0:
        # === CONTROLS OUTSIDE FORM (for navigation) ===
        col_filter1, col_filter2, col_filter3 = st.columns([2, 1, 1])
        
        with col_filter1:
            filter_barcode = st.selectbox(
                "Filtrar por BarCode (opcional)",
                options=['Todos'] + list(selector_df['BarCode'].unique()),
                key='batch_filter_barcode'
            )
        
        with col_filter2:
            # Use selectbox with key - session_state handles the default value
            page_size = st.selectbox(
                "Imágenes por página",
                options=[50, 100, 200],
                key='batch_page_size'
            )
        
        with col_filter3:
            st.metric("Total", len(selector_df))
        
        # Filter data
        if filter_barcode != 'Todos':
            display_df = selector_df[selector_df['BarCode'] == filter_barcode].copy()
        else:
            display_df = selector_df.copy()
        
        # Calculate pagination (use the widget value directly)
        total_images = len(display_df)
        total_pages = max(1, (total_images + st.session_state.batch_page_size - 1) // st.session_state.batch_page_size)  # Ceiling division
        current_page = min(st.session_state.batch_page_number, total_pages)
        
        # Calculate slice for current page
        start_idx = (current_page - 1) * st.session_state.batch_page_size
        end_idx = min(start_idx + st.session_state.batch_page_size, total_images)
        
        # Pagination controls (OUTSIDE FORM - with auto-save)
        st.markdown("---")
        
        # Helper function to save pending changes before navigation
        def save_pending_changes():
            """Save any pending changes from the form before navigating"""
            if st.session_state.batch_pending_changes:
                for row_idx, changes in st.session_state.batch_pending_changes.items():
                    if 'checked' in changes:
                        st.session_state.batch_selector_df.loc[row_idx, '✅'] = changes['checked']
                    if 'tipo' in changes:
                        st.session_state.batch_selector_df.loc[row_idx, 'Tipo'] = changes['tipo']
                
                # Clear pending changes after saving
                st.session_state.batch_pending_changes = {}
        
        page_col1, page_col2, page_col3, page_col4, page_col5 = st.columns([1, 1, 2, 1, 1])
        
        with page_col1:
            # Check if there's active processing
            has_active_processing = len(st.session_state.get('processing_queue', [])) > 0
            
            if st.button("⬅️ Anterior", disabled=(current_page == 1 or has_active_processing), use_container_width=True, key='btn_prev'):
                save_pending_changes()  # Auto-save before navigation
                # Save checkpoint before moving pages
                save_checkpoint_incremental()
                st.session_state.batch_page_number = max(1, current_page - 1)
                st.rerun()
            
            # Show warning if disabled due to processing
            if has_active_processing and current_page > 1:
                st.caption("⏳ Navegación bloqueada: hay procesamiento activo")
        
        with page_col2:
            st.markdown(f"<div style='text-align: center; padding: 8px;'>Página<br/><strong>{current_page}/{total_pages}</strong></div>", unsafe_allow_html=True)
        
        with page_col3:
            # Page selector
            page_number = st.number_input(
                "Ir a página:",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                step=1,
                key='batch_page_selector'
            )
            if page_number != current_page:
                save_pending_changes()  # Auto-save before navigation
                st.session_state.batch_page_number = page_number
                st.rerun()
        
        with page_col4:
            st.markdown(f"<div style='text-align: center; padding: 8px;'>Mostrando<br/><strong>{start_idx + 1}-{end_idx}</strong></div>", unsafe_allow_html=True)
        
        with page_col5:
            # Check if there's active processing
            has_active_processing = len(st.session_state.get('processing_queue', [])) > 0
            
            if st.button("Siguiente ➡️", disabled=(current_page == total_pages or has_active_processing), use_container_width=True, key='btn_next'):
                save_pending_changes()  # Auto-save before navigation
                # Save checkpoint before moving pages
                save_checkpoint_incremental()
                st.session_state.batch_page_number = min(total_pages, current_page + 1)
                st.rerun()
            
            # Show warning if disabled due to processing
            if has_active_processing and current_page < total_pages:
                st.caption("⏳ Navegación bloqueada: hay procesamiento activo")
        
        st.markdown(f"<div style='text-align: center; color: #666; margin: 10px 0;'>Mostrando ({start_idx + 1}-{end_idx})</div>", unsafe_allow_html=True)
        
        # Add reload button for when images don't load properly
        reload_col1, reload_col2, reload_col3 = st.columns([1, 2, 1])
        with reload_col2:
            if st.button("🔄 Recargar Página (si hay problemas de carga)", use_container_width=True, key="reload_page_btn", help="Recarga la página actual sin perder datos si las imágenes no se cargan correctamente"):
                logger.info("🔄 Page reload requested by user")
                st.rerun()
        
        # === HELPER FUNCTION: Count page statistics ===
        def count_page_stats(page_df, selector_df, individual_results_dict):
            """Calculate real-time statistics for current page ONLY
            
            Args:
                page_df: Slice of display_df for current page (preserves original indices)
                selector_df: Full selector DataFrame with all data
                individual_results_dict: Dict with processing results
            
            Returns:
                Tuple: (selected, pending, processing, success) counts for current page
            """
            pending = 0
            processing = 0
            success = 0
            selected = 0
            
            logger.debug(f"count_page_stats: Counting page with {len(page_df)} rows, results_dict has {len(individual_results_dict)} entries")
            
            # IMPORTANT: page_df.index contains ORIGINAL indices from selector_df
            # So we must use row_idx to access selector_df, not sequential idx
            for idx in range(len(page_df)):
                row_data = page_df.iloc[idx]
                row_idx = page_df.index[idx]  # Original index in full DataFrame
                
                barcode = row_data['BarCode']
                tipo_selected = row_data['Tipo']
                img_idx_global = row_data['_img_idx']
                row_key = f"{barcode}_{tipo_selected}_{img_idx_global}"
                
                # Check if checkbox is selected from FULL selector_df using original index
                is_checked = selector_df.loc[row_idx, '✅'] if '✅' in selector_df.columns else False
                
                if is_checked:
                    selected += 1
                    has_result = row_key in individual_results_dict
                    
                    if has_result:
                        status = individual_results_dict[row_key].get('status')
                        if status == 'processing':
                            processing += 1
                        elif status == 'success':
                            success += 1
                        else:
                            pending += 1
                    else:
                        pending += 1
            
            logger.debug(f"count_page_stats RESULT: selected={selected}, pending={pending}, processing={processing}, success={success}")
            return selected, pending, processing, success
        
        def count_global_stats(selector_df, individual_results_dict):
            """Calculate statistics for ALL pages (entire dataset) - CRITICAL for accurate counting"""
            pending = 0
            processing = 0
            success = 0
            selected = 0
            
            logger.debug(f"count_global_stats: Counting all {len(selector_df)} rows, results_dict has {len(individual_results_dict)} entries")
            
            # Iterate through ENTIRE selector_df (all pages)
            for idx in range(len(selector_df)):
                row_data = selector_df.iloc[idx]
                barcode = row_data['BarCode']
                tipo_selected = row_data['Tipo']
                img_idx_global = row_data['_img_idx']
                row_key = f"{barcode}_{tipo_selected}_{img_idx_global}"
                
                # Check if checkbox is selected
                is_checked = row_data['✅'] if '✅' in selector_df.columns else False
                
                if is_checked:
                    selected += 1
                    has_result = row_key in individual_results_dict
                    
                    if has_result:
                        status = individual_results_dict[row_key].get('status')
                        if status == 'processing':
                            processing += 1
                        elif status == 'success':
                            success += 1
                        else:
                            pending += 1
                    else:
                        pending += 1
            
            logger.debug(f"count_global_stats RESULT: selected={selected}, pending={pending}, processing={processing}, success={success}")
            return selected, pending, processing, success
        
        # === BOTÓN PROCESAR PÁGINA COMPLETA ===
        st.markdown("---")
        
        # Pre-calculate page_df
        page_df = display_df.iloc[start_idx:end_idx]
        
        # Calculate BOTH page stats and global stats for accuracy
        selected_count_page, pending_count_page, processing_count_page, success_count_page = count_page_stats(
            page_df, 
            st.session_state.batch_selector_df,
            st.session_state.get('individual_results', {})
        )
        
        # CRITICAL: Calculate GLOBAL stats (all pages) for accurate total count
        selected_count_global, pending_count_global, processing_count_global, success_count_global = count_global_stats(
            st.session_state.batch_selector_df,
            st.session_state.get('individual_results', {})
        )
        
        # Log counts for debugging
        logger.debug(f"📊 PAGE STATS: selected={selected_count_page}, pending={pending_count_page}, processing={processing_count_page}, success={success_count_page}")
        logger.debug(f"📊 GLOBAL STATS: selected={selected_count_global}, pending={pending_count_global}, processing={processing_count_global}, success={success_count_global}")
        logger.debug(f"📊 Current page: {current_page}, indices: {page_df.index.tolist()[:5]}... (showing first 5)")
        
        # Show page processing button and selection controls
        btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
        
        with btn_col1:
            # Select/Deselect all button
            if st.button("✅ Seleccionar Todas", use_container_width=True, key="select_all_btn"):
                # Select all images on current page that aren't already successfully processed
                individual_results = st.session_state.get('individual_results', {})
                selected_count = 0
                skipped_count = 0
                
                for idx in range(len(page_df)):
                    row_data = page_df.iloc[idx]
                    row_idx = page_df.index[idx]
                    barcode = row_data['BarCode']
                    tipo_selected = row_data['Tipo']
                    img_idx_global = row_data['_img_idx']
                    row_key = f"{barcode}_{tipo_selected}_{img_idx_global}"
                    
                    # Check if already processed successfully
                    has_result = row_key in individual_results
                    is_success = has_result and individual_results[row_key].get('status') == 'success'
                    
                    if not is_success:
                        st.session_state.batch_selector_df.loc[row_idx, '✅'] = True
                        selected_count += 1
                        logger.debug(f"Selected row_idx={row_idx}, BarCode={barcode}, row_key={row_key}")
                    else:
                        skipped_count += 1
                        logger.debug(f"Skipped (already processed) row_idx={row_idx}, BarCode={barcode}")
                
                logger.info(f"✅ Selected {selected_count} images on current page (skipped {skipped_count} already processed)")
                save_configuration(st.session_state.batch_selector_df)
                st.session_state['_stats_dirty'] = False
                st.rerun()
        
        # === RESUMEN VISUAL DE PÁGINA (siempre visible) ===
        st.markdown("---")
        
        # Show warning if stats are dirty (changes not yet reflected)
        if st.session_state.get('_stats_dirty', False):
            st.error("⚠️ **CONTADORES DESACTUALIZADOS** - Has realizado cambios en las selecciones. **Haz clic en el botón 🔄 ACTUALIZAR para ver los números correctos.**")
        
        # Show GLOBAL statistics first (most important for user)
        st.markdown("#### 📊 **TOTALES GLOBALES** (Todas las páginas)")
        global_col1, global_col2, global_col3, global_col4, global_col5 = st.columns([1.5, 1.5, 1.5, 1.5, 1])
        
        with global_col1:
            st.metric("✅ TOTAL Seleccionadas", selected_count_global, help="Total de imágenes marcadas en TODAS las páginas")
        with global_col2:
            st.metric("⏳ TOTAL Pendientes", pending_count_global, delta=None if pending_count_global == 0 else "por procesar", delta_color="off")
        with global_col3:
            st.metric("🔄 TOTAL Procesando", processing_count_global, delta=None if processing_count_global == 0 else "en cola", delta_color="off")
        with global_col4:
            st.metric("✅ TOTAL Completadas", success_count_global, delta=None if success_count_global == 0 else "exitosas", delta_color="normal")
        with global_col5:
            # Permanent refresh button - make it prominent when dirty
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            button_type = "primary" if st.session_state.get('_stats_dirty', False) else "secondary"
            button_label = "🔄 ACTUALIZAR" if st.session_state.get('_stats_dirty', False) else "🔄"
            if st.button(button_label, help="Actualizar contadores ahora", use_container_width=True, key="refresh_stats_permanent", type=button_type):
                # Clear dirty flag and force refresh
                st.session_state['_stats_dirty'] = False
                logger.info("🔄 Manual refresh triggered by user")
                st.rerun()
        
        # Show page-specific statistics (secondary)
        st.markdown("##### 📄 Página Actual")
        page_col1, page_col2, page_col3, page_col4 = st.columns([1.5, 1.5, 1.5, 1.5])
        
        with page_col1:
            st.metric("Seleccionadas (página)", f"{selected_count_page}/{len(page_df)}", help="Imágenes marcadas en esta página")
        with page_col2:
            st.metric("Pendientes (página)", pending_count_page, delta_color="off")
        with page_col3:
            st.metric("Procesando (página)", processing_count_page, delta_color="off")
        with page_col4:
            st.metric("Completadas (página)", success_count_page, delta_color="normal")
        
        with btn_col2:
            # Determine button state based on PAGE stats
            is_processing = processing_count_page > 0
            all_processed = pending_count_page == 0 and processing_count_page == 0
            no_selection = selected_count_page == 0
            
            if no_selection:
                st.warning("⚠️ Selecciona al menos una imagen con ✅")
            elif is_processing:
                st.info(f"⏳ Procesando página... ({processing_count_page} imágenes en cola)")
            elif all_processed and selected_count_page > 0:
                st.success(f"✅ Todas las imágenes de esta página ya están procesadas")
            else:
                button_label = f"🚀 Procesar Página Actual ({pending_count_page} pendientes"
                if processing_count_page > 0:
                    button_label += f", {processing_count_page} en cola"
                button_label += ")"
                
                if st.button(
                    button_label,
                    type="primary",
                    use_container_width=True,
                    key="process_page_btn"
                ):
                    # Add only SELECTED pending images from current page to queue
                    logger.info(f"🚀 Starting page processing: {pending_count_page} pending images (from {selected_count_page} selected)")
                    
                    for idx in range(len(page_df)):
                        row_data = page_df.iloc[idx]
                        row_idx = page_df.index[idx]
                        barcode = row_data['BarCode']
                        tipo_selected = row_data['Tipo']
                        img_path = row_data['_img_path']
                        img_idx_global = row_data['_img_idx']  # GLOBAL index across all images
                        row_key = f"{barcode}_{tipo_selected}_{img_idx_global}"
                        
                        # Check if this row is selected
                        is_checked = st.session_state.batch_selector_df.loc[row_idx, '✅'] if '✅' in st.session_state.batch_selector_df.columns else False
                        if not is_checked:
                            continue  # Skip unchecked images
                        
                        # Only add if not already processed or in queue
                        has_result = row_key in st.session_state.get('individual_results', {})
                        if has_result:
                            status = st.session_state.individual_results[row_key].get('status')
                            if status in ['success', 'processing']:
                                continue  # Skip already processed/queued
                        
                        # Add to queue (will auto-number if needed)
                        process_single_image(barcode, img_path, tipo_selected, row_key, row_idx)
                    
                    logger.info(f"✅ All selected pending images added to queue")
                    # Save and rerun to start processing
                    save_checkpoint_incremental()
                    st.session_state['_stats_dirty'] = False
                    st.rerun()
        
        with btn_col3:
            # Deselect all button
            if st.button("❌ Deseleccionar Todas", use_container_width=True, key="deselect_all_btn"):
                # Deselect all images on current page
                deselected_count = 0
                for idx in range(len(page_df)):
                    row_idx = page_df.index[idx]
                    row_data = page_df.iloc[idx]
                    
                    # Log for debugging
                    current_value = st.session_state.batch_selector_df.loc[row_idx, '✅']
                    logger.debug(f"Deselecting row_idx={row_idx}, BarCode={row_data['BarCode']}, current_value={current_value}")
                    
                    st.session_state.batch_selector_df.loc[row_idx, '✅'] = False
                    deselected_count += 1
                
                logger.info(f"✅ Deselected {deselected_count} images on current page (indices: {page_df.index.tolist()})")
                save_configuration(st.session_state.batch_selector_df)
                st.session_state['_stats_dirty'] = False
                st.rerun()
        
        st.markdown("---")
        
        # === IMAGE PROCESSING TABLE WITH INTEGRATED WORKFLOW ===
        
        # Re-get page slice (may have changed after processing)
        page_df = display_df.iloc[start_idx:end_idx]
        
        # Header row - With selection checkbox
        st.markdown("**✅ | 🖼️ Imagen | BarCode | Tipo | 📊 Estado**")
        st.markdown("---")
        
        # Get current page slice
        page_df = display_df.iloc[start_idx:end_idx]
        
        # Group by BarCode for visual separation
        current_barcode = None
        barcode_color_index = 0
        
        # Count images per barcode (for ALL data, not just current page)
        barcode_counts = display_df.groupby('BarCode').size().to_dict()
        barcode_img_counter = {}  # Track which image number we're on for each barcode
        
        for idx in range(len(page_df)):
            row_data = page_df.iloc[idx]
            row_idx = page_df.index[idx]
            barcode = row_data['BarCode']
            
            # Check if we're starting a new BarCode group
            if barcode != current_barcode:
                if current_barcode is not None:
                    # Add separator between groups
                    st.markdown("<div style='margin: 15px 0; border-top: 2px solid #ddd;'></div>", unsafe_allow_html=True)
                
                # Show BarCode group header
                current_barcode = barcode
                barcode_color_index += 1
                barcode_img_counter[barcode] = 0
                
                # Alternating background colors
                bg_color = "#f8f9fa" if barcode_color_index % 2 == 0 else "#ffffff"
                
                st.markdown(f"""
                <div style='background-color: {bg_color}; padding: 8px; border-radius: 5px; margin: 10px 0 5px 0; border-left: 4px solid #0066cc;'>
                    <strong>📦 BarCode: {barcode}</strong> 
                    <span style='color: #666; font-size: 0.9em;'>({barcode_counts[barcode]} imágenes)</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Increment image counter for this barcode
            barcode_img_counter[barcode] += 1
            img_num = barcode_img_counter[barcode]
            total_imgs = barcode_counts[barcode]
            
            # Create unique key for this row using GLOBAL _img_idx
            img_idx_global = row_data['_img_idx']
            current_tipo = row_data['Tipo']
            row_key = f"{barcode}_{current_tipo}_{img_idx_global}"
            result_key = row_key
            
            # Check if this image has been processed
            has_result = result_key in st.session_state.get('individual_results', {})
            result_data = st.session_state.individual_results.get(result_key, {}) if has_result else {}
            
            # Row with integrated workflow - horizontal layout
            bg_color = "#f8f9fa" if barcode_color_index % 2 == 0 else "#ffffff"
            
            st.markdown(f"<div style='background-color: {bg_color}; padding: 10px; border-radius: 5px; margin: 5px 0;'>", unsafe_allow_html=True)
            
            # Main row: Checkbox | Thumbnail | BarCode/Info | Tipo Selector | Status
            row_cols = st.columns([0.5, 1, 1.5, 1.5, 1])
            
            with row_cols[0]:
                # Checkbox for selection
                is_checked = st.session_state.batch_selector_df.loc[row_idx, '✅'] if '✅' in st.session_state.batch_selector_df.columns else False
                
                # Disable checkbox if already processed successfully
                is_disabled = has_result and result_data.get('status') == 'success'
                
                # Create unique callback key for this checkbox
                callback_key = f"_callback_{row_key}"
                
            # Callback function to save changes immediately
            if callback_key not in st.session_state:
                def make_callback(r_idx, r_key):
                    def callback():
                        # Get new value from widget
                        new_value = st.session_state[f"check_{r_key}"]
                        # Get old value to detect actual changes
                        old_value = st.session_state.batch_selector_df.loc[r_idx, '✅']
                        # Save immediately to DataFrame
                        st.session_state.batch_selector_df.loc[r_idx, '✅'] = new_value
                        save_configuration(st.session_state.batch_selector_df)
                        # Mark stats as dirty ONLY if value actually changed
                        if new_value != old_value:
                            st.session_state['_stats_dirty'] = True
                            logger.debug(f"Checkbox callback: row_idx={r_idx}, changed {old_value} -> {new_value}, marked dirty")
                        else:
                            logger.debug(f"Checkbox callback: row_idx={r_idx}, no change (value={new_value})")
                    return callback
                
                st.session_state[callback_key] = make_callback(row_idx, row_key)
                
                checked = st.checkbox(
                    "Sel",
                    value=is_checked,
                    key=f"check_{row_key}",
                    label_visibility="collapsed",
                    disabled=is_disabled,
                    on_change=st.session_state[callback_key]
                )
            
            with row_cols[1]:
                # Image thumbnail with caching and lazy loading
                try:
                    img_path = row_data['_img_path']
                    
                    # Check if file exists before attempting to load
                    if not Path(img_path).exists():
                        st.caption("❌ No encontrada")
                        logger.warning(f"Image file not found: {img_path}")
                    else:
                        try:
                            # Load and resize image to thumbnail size for faster display
                            with Image.open(img_path) as img:
                                # Calculate thumbnail size maintaining aspect ratio
                                max_width = 80
                                aspect_ratio = img.height / img.width
                                thumbnail_height = int(max_width * aspect_ratio)
                                
                                # Resize using LANCZOS for quality
                                img_thumbnail = img.resize((max_width, thumbnail_height), Image.Resampling.LANCZOS)
                                
                                # Convert to RGB if needed (handles RGBA, grayscale, etc.)
                                if img_thumbnail.mode != 'RGB':
                                    img_thumbnail = img_thumbnail.convert('RGB')
                            
                            # Display thumbnail
                            st.image(img_thumbnail, use_container_width=False)
                        except Exception as thumb_error:
                            # Fallback to direct display if thumbnail creation fails
                            logger.debug(f"Thumbnail creation failed for {img_path}, using direct display: {thumb_error}")
                            st.image(img_path, width=80)
                    
                    # Button to view full size
                    if st.button("🔍", key=f"zoom_{row_key}", help="Ver en tamaño completo", use_container_width=True):
                        st.session_state[f'show_fullsize_{row_key}'] = True
                        
                except Exception as e:
                    st.caption("❌ Error")
                    logger.error(f"Error loading image thumbnail for {barcode}: {e}, path: {row_data.get('_img_path', 'N/A')}")
            
            with row_cols[2]:
                st.caption(f"**{barcode}**")
                st.caption(f"Img {img_num}/{total_imgs}")
            
            # Show full-size image in dialog if requested
            if st.session_state.get(f'show_fullsize_{row_key}', False):
                with st.expander(f"🖼️ Vista Completa: {barcode} - Img {img_num}", expanded=True):
                    try:
                        # Show full-size image
                        st.image(row_data['_img_path'], use_container_width=True, caption=f"{barcode} - Imagen {img_num}/{total_imgs}")
                        
                        # Close button
                        if st.button("✖️ Cerrar", key=f"close_zoom_{row_key}"):
                            st.session_state[f'show_fullsize_{row_key}'] = False
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error cargando imagen: {e}")
            
            with row_cols[3]:
                # Type selector with radio buttons
                current_type = row_data['Tipo']
                tipo_options = ['PLACA1', 'PLACA2', 'SCADA1', 'SCADA2', 'AMBOS']
                
                tipo_selected = st.radio(
                    "Tipo",
                    options=tipo_options,
                    index=tipo_options.index(current_type) if current_type in tipo_options else 0,
                    key=f"tipo_sel_{row_key}",
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                # Update in dataframe if changed
                if tipo_selected != current_type:
                    st.session_state.batch_selector_df.loc[row_idx, 'Tipo'] = tipo_selected
                    save_configuration(st.session_state.batch_selector_df)
            
            with row_cols[4]:
                # Show status
                if has_result:
                    status = result_data.get('status')
                    if status == 'processing':
                        # Check if it's first in queue (actively processing) or waiting
                        queue_pos = st.session_state.processing_queue.index(row_key) + 1 if row_key in st.session_state.processing_queue else 0
                        if queue_pos == 1:
                            st.info("🔄 Procesando...")
                        elif queue_pos > 1:
                            st.warning(f"⏳ Cola #{queue_pos}")
                        else:
                            st.info("⏳ Cola")
                    elif status == 'success':
                        proc_time = result_data.get('processing_time', 0)
                        st.success(f"✅ {proc_time:.1f}s")
                    elif status == 'error':
                        st.error("❌")
                else:
                    st.caption("⚪ Pendiente")
            
            # Close the colored row div before showing expander
            st.markdown("</div>", unsafe_allow_html=True)
            
            # If processed, show results in collapsible section (OUTSIDE columns, full width)
            if has_result and result_data.get('status') == 'success':
                with st.expander(f"📊 Resultados y Corrección - {barcode} [Img {img_num}]", expanded=True):
                    api_data = result_data.get('data', {})
                    
                    # Show extraction info
                    info_col1, info_col2, info_col3 = st.columns(3)
                    with info_col1:
                        st.caption(f"**Método:** {api_data.get('method', 'API')}")
                    with info_col2:
                        st.caption(f"**Modelo:** {api_data.get('model', 'N/A')}")
                    with info_col3:
                        conf = api_data.get('overall_confidence', 0.0)
                        conf_color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
                        st.caption(f"**Confianza:** {conf_color} {conf:.1%}")
                    
                    st.markdown("---")
                    
                    # Initialize edited values in session state if not exists
                    edit_key = f"edit_{row_key}"
                    if edit_key not in st.session_state:
                        st.session_state[edit_key] = {
                            'marca': api_data.get('marca', ''),
                            'modelo': api_data.get('modelo', ''),
                            'numero_serie': api_data.get('numero_serie', ''),
                            'año': api_data.get('año', ''),
                            'potencia': api_data.get('potencia', ''),
                            'codigo_scada_principal': api_data.get('codigo_scada_principal', '')
                        }
                    
                    # Show editable fields based on type
                    if 'PLACA' in tipo_selected or tipo_selected == 'AMBOS':
                        st.caption("**Datos de Placa Técnica**")
                        col_edit1, col_edit2, col_edit3 = st.columns(3)
                        with col_edit1:
                            marca_edit = st.text_input(
                                "Marca",
                                value=st.session_state[edit_key]['marca'] or '',
                                key=f"marca_edit_{row_key}",
                                placeholder="Ej: SCHNEIDER"
                            )
                            if marca_edit != st.session_state[edit_key].get('marca'):
                                st.session_state.pending_edits.add(row_key)
                            st.session_state[edit_key]['marca'] = marca_edit
                            
                            modelo_edit = st.text_input(
                                "Modelo",
                                value=st.session_state[edit_key]['modelo'] or '',
                                key=f"modelo_edit_{row_key}",
                                placeholder="Ej: XYZ-100"
                            )
                            st.session_state[edit_key]['modelo'] = modelo_edit
                        
                        with col_edit2:
                            sn_edit = st.text_input(
                                "Número Serie",
                                value=st.session_state[edit_key]['numero_serie'] or '',
                                key=f"sn_edit_{row_key}",
                                placeholder="Ej: ABC123456"
                            )
                            st.session_state[edit_key]['numero_serie'] = sn_edit
                            
                            año_edit = st.text_input(
                                "Año",
                                value=str(st.session_state[edit_key]['año']) if st.session_state[edit_key]['año'] else '',
                                key=f"año_edit_{row_key}",
                                placeholder="Ej: 2020"
                            )
                            st.session_state[edit_key]['año'] = año_edit
                        
                        with col_edit3:
                            potencia_edit = st.text_input(
                                "Potencia",
                                value=st.session_state[edit_key]['potencia'] or '',
                                key=f"potencia_edit_{row_key}",
                                placeholder="Ej: 5.5KW"
                            )
                            st.session_state[edit_key]['potencia'] = potencia_edit
                    
                    if 'SCADA' in tipo_selected or tipo_selected == 'AMBOS':
                        st.caption("**Código SCADA**")
                        scada_edit = st.text_input(
                            "Código SCADA Principal",
                            value=st.session_state[edit_key]['codigo_scada_principal'] or '',
                            key=f"scada_edit_{row_key}",
                            placeholder="Ej: AB-1234.567"
                        )
                        st.session_state[edit_key]['codigo_scada_principal'] = scada_edit
                    
                    # Save button for edits
                    if st.button("💾 Guardar Correcciones", key=f"save_edit_{row_key}", use_container_width=True):
                        # Update the result with edited values
                        result_data['data'].update(st.session_state[edit_key])
                        st.session_state.individual_results[result_key] = result_data
                        
                        # Remove from pending edits
                        st.session_state.pending_edits.discard(row_key)
                        
                        # Update batch_results (avoid duplicates)
                        found = False
                        for batch_result in st.session_state.batch_results:
                            if batch_result['BarCode'] == barcode:
                                found = True
                                # Update or create tipo entry
                                if 'results_by_type' not in batch_result:
                                    batch_result['results_by_type'] = {}
                                batch_result['results_by_type'][tipo_selected] = st.session_state[edit_key].copy()
                                batch_result['timestamp'] = datetime.now().isoformat()
                                break
                        
                        if not found:
                            # Create new batch entry if barcode doesn't exist
                            st.session_state.batch_results.append({
                                'BarCode': barcode,
                                'results_by_type': {tipo_selected: st.session_state[edit_key].copy()},
                                'timestamp': datetime.now().isoformat()
                            })
                        
                        # Auto-save to disk (deduplicate by BarCode)
                        try:
                            output_dir = Path(st.session_state.batch_config.output_dir) / "batch_results"
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Use consistent filename for incremental updates
                            json_file = output_dir / "corrected_results_latest.json"
                            
                            # Deduplicate batch_results before saving
                            unique_results = {}
                            for batch_result in st.session_state.batch_results:
                                bc = batch_result['BarCode']
                                if bc not in unique_results:
                                    unique_results[bc] = batch_result
                                else:
                                    # Merge results_by_type
                                    unique_results[bc]['results_by_type'].update(batch_result.get('results_by_type', {}))
                                    unique_results[bc]['timestamp'] = batch_result.get('timestamp', datetime.now().isoformat())
                            
                            # Convert back to list
                            deduplicated_results = list(unique_results.values())
                            
                            with open(json_file, 'w', encoding='utf-8') as f:
                                json.dump(deduplicated_results, f, indent=2, ensure_ascii=False, default=str)
                            
                            st.success("✅ Correcciones guardadas")
                            logger.info(f"Corrections saved for {barcode} [{tipo_selected}] - Total unique results: {len(deduplicated_results)}")
                        except Exception as e:
                            st.error(f"Error guardando: {e}")
                            logger.error(f"Failed to save corrections: {e}")
            
            elif has_result and result_data.get('status') == 'error':
                st.error(f"❌ Error: {result_data.get('error', 'Unknown')}")
        
        # Info message about total items
        if total_pages > 1:
            st.info(f"💡 Navegando página {current_page} de {total_pages}. Total: {total_images} imágenes")
        
        st.markdown("---")
        
        # === PAGINATION CONTROLS AT BOTTOM ===
        if len(selector_df) > 0 and total_pages > 1:
            st.markdown("---")
            st.markdown("#### 📄 Navegación de Páginas")
            
            page_col1_btm, page_col2_btm, page_col3_btm, page_col4_btm, page_col5_btm = st.columns([1, 1, 2, 1, 1])
            
            with page_col1_btm:
                # Check if there's active processing
                has_active_processing = len(st.session_state.get('processing_queue', [])) > 0
                
                if st.button("⬅️ Anterior", disabled=(current_page == 1 or has_active_processing), use_container_width=True, key='btn_prev_bottom'):
                    # Warn about unsaved edits
                    if st.session_state.pending_edits:
                        st.warning(f"⚠️ Tienes {len(st.session_state.pending_edits)} edición(es) sin guardar. Se guardarán automáticamente.")
                        # Auto-save all pending edits
                        try:
                            output_dir = Path(st.session_state.batch_config.output_dir) / "batch_results"
                            output_dir.mkdir(parents=True, exist_ok=True)
                            json_file = output_dir / "corrected_results_latest.json"
                            
                            # Deduplicate before saving
                            unique_results = {}
                            for batch_result in st.session_state.batch_results:
                                bc = batch_result['BarCode']
                                if bc not in unique_results:
                                    unique_results[bc] = batch_result
                                else:
                                    unique_results[bc]['results_by_type'].update(batch_result.get('results_by_type', {}))
                            
                            with open(json_file, 'w', encoding='utf-8') as f:
                                json.dump(list(unique_results.values()), f, indent=2, ensure_ascii=False, default=str)
                            
                            st.session_state.pending_edits.clear()
                            logger.info("Auto-saved pending edits before page change")
                        except Exception as e:
                            logger.error(f"Failed to auto-save: {e}")
                    
                    save_pending_changes()  # Auto-save configuration
                    st.session_state.batch_page_number = max(1, current_page - 1)
                    st.rerun()
            
            with page_col2_btm:
                st.markdown(f"<div style='text-align: center; padding: 8px;'>Página<br/><strong>{current_page}/{total_pages}</strong></div>", unsafe_allow_html=True)
            
            with page_col3_btm:
                # Page selector
                page_number_btm = st.number_input(
                    "Ir a página:",
                    min_value=1,
                    max_value=total_pages,
                    value=current_page,
                    step=1,
                    key='batch_page_selector_bottom'
                )
                if page_number_btm != current_page:
                    save_pending_changes()  # Auto-save before navigation
                    st.session_state.batch_page_number = page_number_btm
                    st.rerun()
            
            with page_col4_btm:
                st.markdown(f"<div style='text-align: center; padding: 8px;'>Mostrando<br/><strong>{start_idx + 1}-{end_idx}</strong></div>", unsafe_allow_html=True)
            
            with page_col5_btm:
                # Check if there's active processing
                has_active_processing = len(st.session_state.get('processing_queue', [])) > 0
                
                if st.button("Siguiente ➡️", disabled=(current_page == total_pages or has_active_processing), use_container_width=True, key='btn_next_bottom'):
                    # Warn about unsaved edits
                    if st.session_state.pending_edits:
                        st.warning(f"⚠️ Tienes {len(st.session_state.pending_edits)} edición(es) sin guardar. Se guardarán automáticamente.")
                        # Auto-save all pending edits
                        try:
                            output_dir = Path(st.session_state.batch_config.output_dir) / "batch_results"
                            output_dir.mkdir(parents=True, exist_ok=True)
                            json_file = output_dir / "corrected_results_latest.json"
                            
                            # Deduplicate before saving
                            unique_results = {}
                            for batch_result in st.session_state.batch_results:
                                bc = batch_result['BarCode']
                                if bc not in unique_results:
                                    unique_results[bc] = batch_result
                                else:
                                    unique_results[bc]['results_by_type'].update(batch_result.get('results_by_type', {}))
                            
                            with open(json_file, 'w', encoding='utf-8') as f:
                                json.dump(list(unique_results.values()), f, indent=2, ensure_ascii=False, default=str)
                            
                            st.session_state.pending_edits.clear()
                            logger.info("Auto-saved pending edits before page change")
                        except Exception as e:
                            logger.error(f"Failed to auto-save: {e}")
                    
                    save_pending_changes()  # Auto-save configuration
                    st.session_state.batch_page_number = min(total_pages, current_page + 1)
                    st.rerun()
    
    # Use the updated dataframe
    edited_df = st.session_state.batch_selector_df.copy()
    
    # Save edited data back to session state
    st.session_state.batch_selector_df = edited_df
    
    # ==== CONVERT TO BATCH_SELECTIONS FORMAT ====
    # Update batch_selections from edited dataframe for processing
    st.session_state.batch_selections = {}
    
    for _, row in edited_df.iterrows():
        if row['✅']:  # Updated column name
            barcode = row['_barcode_orig']
            img_idx = row['_img_idx']
            img_path = row['_img_path']
            tipo_raw = row['Tipo'].upper().strip()
            
            # Determine type from user input (PLACA1, PLACA2, SCADA1, AMBOS, etc.)
            if 'PLACA' in tipo_raw:
                tipo = 'placa_tecnica'
            elif 'SCADA' in tipo_raw:
                tipo = 'codigo_scada'
            elif tipo_raw == 'AMBOS':
                tipo = None  # None triggers auto-detection (extracts all fields)
            else:
                # Default to auto-detect if unclear
                tipo = None
            
            if barcode not in st.session_state.batch_selections:
                st.session_state.batch_selections[barcode] = {}
            
            # Use the full tipo_raw as key to support PLACA1, PLACA2, etc.
            st.session_state.batch_selections[barcode][img_idx] = {
                'process': True,
                'type': tipo,  # Generic type for API
                'type_label': tipo_raw,  # Full label (PLACA1, SCADA2, etc.)
                'path': img_path
            }
    
    # ==== RESUMEN Y VALIDACIÓN ====
    st.markdown("---")
    st.markdown("### 📊 Resumen Global (Todas las Páginas)")
    st.caption("💡 Este resumen incluye TODAS las imágenes seleccionadas en todas las páginas, no solo la actual")

    # Calculate totals from edited data
    selected_rows = edited_df[edited_df['✅'] == True]
    
    # Filter out already processed images
    pending_rows = []
    processed_count = 0
    
    for _, row in selected_rows.iterrows():
        # Create row_key using GLOBAL _img_idx (critical for multi-page processing)
        # Fallback to 'img_idx' for backward compatibility with old checkpoints
        img_idx_value = row.get('_img_idx', row.get('img_idx', 0))
        row_key = f"{row['BarCode']}_{row['Tipo']}_{img_idx_value}"
        
        # Check if already processed successfully
        individual_results = st.session_state.get('individual_results', {})
        is_processed = (row_key in individual_results and 
                       individual_results[row_key].get('status') == 'success')
        
        if is_processed:
            processed_count += 1
            logger.debug(f"✓ Skipping {row_key} - already processed successfully")
        else:
            pending_rows.append(row)
            logger.debug(f"  Pending: {row_key}")
    
    total_selected = len(selected_rows)
    total_pending = len(pending_rows)
    
    # Count by type prefix (PLACA*, SCADA*) - only pending
    pending_df = pd.DataFrame(pending_rows) if pending_rows else pd.DataFrame()
    if not pending_df.empty:
        total_placa_pending = len(pending_df[pending_df['Tipo'].str.upper().str.contains('PLACA', na=False)])
        total_scada_pending = len(pending_df[pending_df['Tipo'].str.upper().str.contains('SCADA', na=False)])
    else:
        total_placa_pending = 0
        total_scada_pending = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Seleccionadas", total_selected)
        if processed_count > 0:
            st.caption(f"✅ {processed_count} ya procesadas")
    with col2:
        st.metric("📋 Placas Pendientes", total_placa_pending)
    with col3:
        st.metric("🔢 SCADA Pendientes", total_scada_pending)
    with col4:
        if default_model and available_models:
            # Calculate estimated cost for PENDING images only
            selected_model_info = next((m for m in available_models if m['id'] == default_model), None)
            if selected_model_info:
                cost_per_image = selected_model_info.get('cost_per_1k_tokens', 0) * 1000
                total_cost = cost_per_image * total_pending
                st.metric("💰 Costo Est.", f"${total_cost:.4f}")

    # Tips for users
    if total_pending == 0 and total_selected > 0:
        st.success(f"✅ **Todas las {total_selected} imágenes ya están procesadas!** No hay nada nuevo que procesar.", icon="✅")
    elif total_pending == 0:
        st.info("💡 **Tip:** Usa los botones de 'Acciones Rápidas' para marcar todas de una vez, o marca checkboxes individuales en la tabla.", icon="💡")
    elif total_pending > 100:
        st.warning(f"⚠️ Procesarás **{total_pending} imágenes pendientes** ({processed_count} ya procesadas se saltarán automáticamente). Esto puede tomar varios minutos.", icon="⚠️")
    elif processed_count > 0:
        st.info(f"ℹ️ Se procesarán **{total_pending} imágenes nuevas** (el sistema saltará {processed_count} ya procesadas).", icon="ℹ️")    # ==== PREVISUALIZACIÓN DE IMÁGENES SELECCIONADAS ====
    if total_pending > 0:
        with st.expander(f"🖼️ Vista Previa: Imágenes Pendientes ({total_pending} de {total_selected} seleccionadas)", expanded=False):
            st.caption(f"Mostrando las primeras 20 imágenes que se procesarán (excluye {processed_count} ya procesadas)")
            
            # Display pending images (limit to first 20 for performance)
            pending_preview = pending_df.head(20) if not pending_df.empty else pd.DataFrame()
            
            if len(pending_preview) > 0:
                # Display in grid (4 columns)
                num_cols = 4
                for start_idx in range(0, len(pending_preview), num_cols):
                    cols = st.columns(num_cols)
                    
                    for col_offset in range(num_cols):
                        idx = start_idx + col_offset
                        if idx >= len(pending_preview):
                            break
                        
                        row = pending_preview.iloc[idx]
                        
                        with cols[col_offset]:
                            try:
                                img = Image.open(row['_img_path'])
                                st.image(img, use_container_width=True)
                                st.caption(f"{row['BarCode']} - #{row['Img']}")
                                st.caption(f"{row['Tipo']}")
                            except Exception as e:
                                st.error(f"Error: {e}")
                
                if len(pending_preview) == 20 and total_pending > 20:
                    st.info(f"Mostrando 20 de {total_pending} imágenes pendientes")
    
    return total_pending, default_method, default_model, processed_count


def process_single_image(barcode: str, img_path: str, tipo_label: str, row_key: str, row_idx: int):
    """
    Process a single image immediately and update results
    Uses simplified approach with st.rerun() for UI updates
    
    Args:
        barcode: BarCode identifier
        img_path: Path to image file
        tipo_label: Type label (PLACA1, SCADA1, etc.)
        row_key: Unique row identifier
        row_idx: DataFrame index
    """
    logger.info(f"🎯 process_single_image() CALLED: {barcode} [{tipo_label}] from {img_path}")
    
    # Initialize results dict if not exists
    if 'individual_results' not in st.session_state:
        st.session_state.individual_results = {}
        logger.info("Created individual_results dict")
    
    # Check if already processed successfully
    result_key = row_key
    if result_key in st.session_state.individual_results:
        existing = st.session_state.individual_results[result_key]
        if existing.get('status') == 'success':
            st.warning(f"⚠️ {barcode} [{tipo_label}] ya fue procesado exitosamente. No se volverá a procesar.")
            return
    
    # Check if already in queue
    if row_key in st.session_state.processing_queue:
        position = st.session_state.processing_queue.index(row_key) + 1
        st.info(f"ℹ️ {barcode} [{tipo_label}] ya está en la cola (posición #{position})")
        return
    
    # AUTO-NUMBERING: Check if this barcode already has images of this type
    # Extract base type (remove existing _N suffix if present)
    base_type = tipo_label.split('_')[0] if '_' in tipo_label else tipo_label
    
    # Count existing images with same base type for this barcode
    existing_count = 0
    if 'batch_results' in st.session_state:
        for result in st.session_state.batch_results:
            if result['BarCode'] == barcode:
                # Count how many tipo_labels start with this base_type
                for existing_tipo in result.get('results_by_type', {}).keys():
                    if existing_tipo.startswith(base_type):
                        existing_count += 1
                break
    
    # Also check items already in queue for this barcode
    for queued_key in st.session_state.processing_queue:
        if queued_key in st.session_state.individual_results:
            queued_info = st.session_state.individual_results[queued_key]
            if queued_info['barcode'] == barcode:
                queued_tipo = queued_info['tipo_label'].split('_')[0] if '_' in queued_info['tipo_label'] else queued_info['tipo_label']
                if queued_tipo == base_type:
                    existing_count += 1
    
    # Generate auto-numbered tipo_label
    auto_numbered_tipo = f"{base_type}_{existing_count + 1}"
    
    logger.info(f"🔢 Auto-numbering: {tipo_label} -> {auto_numbered_tipo} (found {existing_count} existing)")
    
    # Mark as processing and store all needed info with auto-numbered tipo
    st.session_state.processing_queue.append(row_key)
    
    # Extract img_idx from row_key (last component after final underscore)
    img_idx_str = row_key.split('_')[-1] if '_' in row_key else '0'
    
    st.session_state.individual_results[result_key] = {
        'status': 'processing',
        'barcode': barcode,
        'tipo_label': auto_numbered_tipo,  # Use auto-numbered version
        'original_tipo': tipo_label,  # Keep original for reference
        'img_idx': img_idx_str,  # Store for checkpoint reconstruction
        'img_path': img_path  # Store path here so we don't need to search later
    }
    
    # Log queue addition with detailed info
    logger.info(f"✅ ADDED TO QUEUE: {barcode} [{auto_numbered_tipo}]")
    logger.info(f"   Queue length NOW: {len(st.session_state.processing_queue)}")
    logger.info(f"   Queue contents: {st.session_state.processing_queue}")
    logger.info(f"   Individual results keys: {list(st.session_state.individual_results.keys())}")
    
    # Show immediate feedback with auto-numbered tipo
    logger.info(f"✅ {barcode} [{auto_numbered_tipo}] agregado a la cola (posición #{len(st.session_state.processing_queue)})")
    
    # NOTE: NO rerun here - batch processing will trigger one rerun after all images added
    # This prevents multiple reruns when processing entire page


def process_queued_items():
    """
    Process ALL items from the queue sequentially (called on each page load)
    Shows progress bar and processes one by one
    """
    # Ensure processing_queue exists
    if 'processing_queue' not in st.session_state:
        st.session_state.processing_queue = []
        logger.warning("processing_queue was not initialized, created empty list")
    
    if not st.session_state.processing_queue:
        logger.debug("Queue is empty, nothing to process")
        return
    
    total_items = len(st.session_state.processing_queue)
    logger.info(f"🔍 Queue check: {total_items} items waiting to process")
    
    # Create enhanced progress container with detailed feedback
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🚀 Procesamiento en Curso")
        st.info(f"📋 **Total de imágenes en cola:** {total_items}")
        
        # Progress bar with percentage
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Current image being processed
        status_text = st.empty()
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            time_elapsed_text = st.empty()
        with col2:
            time_remaining_text = st.empty()
        with col3:
            speed_text = st.empty()
        
        # Queue preview (next items)
        queue_preview = st.expander("📜 Ver cola de procesamiento", expanded=False)
    
    start_time = time.time()
    
    # Process each item in queue
    for item_idx in range(total_items):
        if not st.session_state.processing_queue:
            break  # Queue empty (shouldn't happen)
        
        # Calculate progress
        progress_pct = item_idx / total_items
        progress_bar.progress(progress_pct)
        progress_text.markdown(f"**Progreso:** {item_idx}/{total_items} ({progress_pct*100:.1f}%)")
        
        # Calculate time metrics
        elapsed = time.time() - start_time
        if item_idx > 0:
            avg_time_per_image = elapsed / item_idx
            remaining_images = total_items - item_idx
            estimated_remaining = avg_time_per_image * remaining_images
            
            time_elapsed_text.metric("⏱️ Tiempo Transcurrido", f"{elapsed:.1f}s")
            time_remaining_text.metric("🕒 Tiempo Restante", f"{estimated_remaining:.0f}s")
            speed_text.metric("⚡ Velocidad", f"{avg_time_per_image:.1f}s/img")
        else:
            time_elapsed_text.metric("⏱️ Tiempo Transcurrido", "0s")
            time_remaining_text.metric("🕒 Tiempo Restante", "Calculando...")
            speed_text.metric("⚡ Velocidad", "Calculando...")
        
        # Update queue preview
        with queue_preview:
            # Show current item
            current_key = st.session_state.processing_queue[0]
            current_info = st.session_state.individual_results.get(current_key, {})
            st.markdown(f"🔵 **Procesando ahora:** `{current_info.get('barcode', 'N/A')}` [{current_info.get('tipo_label', 'N/A')}]")
            
            # Show next 3 items
            if len(st.session_state.processing_queue) > 1:
                st.markdown("**⏬ Siguientes en cola:**")
                for i in range(1, min(4, len(st.session_state.processing_queue))):
                    next_key = st.session_state.processing_queue[i]
                    next_info = st.session_state.individual_results.get(next_key, {})
                    st.caption(f"  {i}. `{next_info.get('barcode', 'N/A')}` [{next_info.get('tipo_label', 'N/A')}]")
        
        # Process one item
        _process_single_queued_item(status_text, item_idx + 1, total_items)
        
        # Save checkpoint after each image
        save_checkpoint_incremental()
    
    # Final progress
    progress_bar.progress(1.0)
    total_time = time.time() - start_time
    progress_text.markdown(f"**Progreso:** {total_items}/{total_items} (100%)")
    status_text.success(f"✅ ¡Procesamiento completo! {total_items} imágenes procesadas en {total_time:.1f}s")
    
    # Update final metrics
    time_elapsed_text.metric("⏱️ Tiempo Total", f"{total_time:.1f}s")
    time_remaining_text.metric("🕒 Tiempo Restante", "0s")
    speed_text.metric("⚡ Velocidad Promedio", f"{total_time/total_items:.1f}s/img")
    
    time.sleep(1.5)  # Show completion message briefly
    
    # Clear dirty flag after processing - counters should now be accurate
    st.session_state['_stats_dirty'] = False
    logger.info(f"✅ Queue processing complete - stats are now clean")


def _process_single_queued_item(status_text, item_num, total_items):
    """
    Process one item from the queue (internal helper)
    """
    if not st.session_state.processing_queue:
        return
    
    # Get first item in queue
    row_key = st.session_state.processing_queue[0]
    logger.info(f"=== Processing queue item {item_num}/{total_items}: {row_key} ===")
    logger.info(f"Queue length BEFORE processing: {len(st.session_state.processing_queue)}")
    
    # Get the stored processing info
    if row_key not in st.session_state.individual_results:
        # Item was removed, skip
        logger.warning(f"Item {row_key} not found in results, removing from queue")
        st.session_state.processing_queue.pop(0)
        return
    
    result_info = st.session_state.individual_results[row_key]
    if result_info.get('status') != 'processing':
        # Already completed or errored, remove from queue
        logger.warning(f"Item {row_key} status is {result_info.get('status')}, not 'processing'. Removing from queue.")
        st.session_state.processing_queue.pop(0)
        return
    
    barcode = result_info['barcode']
    tipo_label = result_info['tipo_label']
    img_path = result_info.get('img_path')
    
    logger.info(f"Processing: {barcode} [{tipo_label}]")
    logger.info(f"Image path: {img_path}")
    
    # Update status text with detailed info
    status_text.info(f"🔄 **{item_num}/{total_items}:** Procesando `{barcode}` - **{tipo_label}** (usando API OpenAI GPT-4o-mini)")
    
    if not img_path:
        # Can't find image, mark as error
        logger.error(f"Image path not found for {row_key}")
        st.session_state.individual_results[row_key] = {
            'status': 'error',
            'barcode': barcode,
            'tipo_label': tipo_label,
            'error': 'Image path not found'
        }
        st.session_state.processing_queue.pop(0)
        return
    
    try:
        # Get API config
        config_obj = st.session_state.batch_config
        api_config = config_obj.config.get('api', {}).copy() if hasattr(config_obj, 'config') else {}
        
        # Initialize API extractor
        api_extractor = APIExtractor(api_config)
        
        # Map tipo_label to API image_type (handle auto-numbered suffixes)
        # Extract base type (remove _N suffix): PLACA1_2 -> PLACA1
        base_tipo = tipo_label.split('_')[0] if '_' in tipo_label else tipo_label
        
        tipo_mapping = {
            'PLACA1': 'placa_tecnica',
            'PLACA2': 'placa_tecnica',
            'PLACA3': 'placa_tecnica',
            'SCADA1': 'codigo_scada',
            'SCADA2': 'codigo_scada',
            'SCADA3': 'codigo_scada',
            'AMBOS': None  # Auto-detect
        }
        image_type = tipo_mapping.get(base_tipo, None)
        
        logger.info(f"Type mapping: {tipo_label} (base: {base_tipo}) -> {image_type}")
        
        # Process image
        start_time = time.time()
        extraction_result = api_extractor.extract_from_image_assisted(
            img_path,
            barcode,
            preprocessed_path=None,
            image_type=image_type
        )
        processing_time = time.time() - start_time
        
        # Store result
        st.session_state.individual_results[row_key] = {
            'status': 'success',
            'barcode': barcode,
            'tipo_label': tipo_label,
            'img_idx': row_key.split('_')[-1] if '_' in row_key else '0',  # Extract from row_key
            'img_path': img_path,  # Store for matching
            'data': extraction_result,
            'processing_time': processing_time
        }
        
        # Add to batch results for consolidation
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = []
        
        # Check if barcode already exists in batch_results
        existing_idx = None
        for i, result in enumerate(st.session_state.batch_results):
            if result['BarCode'] == barcode:
                existing_idx = i
                break
        
        if existing_idx is not None:
            # Update existing entry - SMART MERGE with intelligent fusion
            existing_result = st.session_state.batch_results[existing_idx]
            
            # Get existing data for this tipo_label (if any)
            existing_data = existing_result['results_by_type'].get(tipo_label, {})
            
            # New data from current processing
            new_data = {
                'marca': extraction_result.get('marca'),
                'modelo': extraction_result.get('modelo'),
                'numero_serie': extraction_result.get('numero_serie'),
                'año': extraction_result.get('año'),
                'potencia': extraction_result.get('potencia'),
                'codigo_scada_principal': extraction_result.get('codigo_scada_principal'),
                'confidence': extraction_result.get('overall_confidence', 0.0)
            }
            
            # INTELLIGENT FUSION: Prefer non-null values
            # If existing has a value and new is null, keep existing
            # If existing is null and new has a value, use new
            # If both have values, use new (latest processing)
            merged_data = {}
            for key in new_data.keys():
                existing_value = existing_data.get(key)
                new_value = new_data[key]
                
                # Fusion logic: prioritize non-null values
                if new_value is not None and new_value != '':
                    # New data has a value, use it
                    merged_data[key] = new_value
                elif existing_value is not None and existing_value != '':
                    # Keep existing value if new is null/empty
                    merged_data[key] = existing_value
                else:
                    # Both null, keep null
                    merged_data[key] = None
            
            # Store merged data
            existing_result['results_by_type'][tipo_label] = merged_data
            
            logger.info(f"SMART MERGE for {barcode} [{tipo_label}]:")
            logger.info(f"  Existing: {existing_data}")
            logger.info(f"  New: {new_data}")
            logger.info(f"  Merged: {merged_data}")
            
            # Add image info (always append to track all processed images)
            if 'images_processed' not in existing_result:
                existing_result['images_processed'] = []
            existing_result['images_processed'].append({
                'image_path': img_path,
                'type': image_type or 'auto',
                'type_label': tipo_label,
                'result': extraction_result
            })
        else:
            # Create new entry
            new_result = {
                'BarCode': barcode,
                'images_processed': [{
                    'image_path': img_path,
                    'type': image_type or 'auto',
                    'type_label': tipo_label,
                    'result': extraction_result
                }],
                'results_by_type': {
                    tipo_label: {
                        'marca': extraction_result.get('marca'),
                        'modelo': extraction_result.get('modelo'),
                        'numero_serie': extraction_result.get('numero_serie'),
                        'año': extraction_result.get('año'),
                        'potencia': extraction_result.get('potencia'),
                        'codigo_scada_principal': extraction_result.get('codigo_scada_principal'),
                        'confidence': extraction_result.get('overall_confidence', 0.0)
                    }
                },
                'processing_time': processing_time,
                'status': 'success',
                'method': 'API OpenAI',
                'model': api_config.get('model', 'gpt-4o-mini-2024-07-18')
            }
            st.session_state.batch_results.append(new_result)
        
        logger.info(f"Successfully processed: {barcode} [{tipo_label}] in {processing_time:.1f}s")
        
        # Show success summary with extracted data
        if 'PLACA' in tipo_label:
            marca = extraction_result.get('marca', 'N/A')
            modelo = extraction_result.get('modelo', 'N/A')
            potencia = extraction_result.get('potencia', 'N/A')
            status_text.success(f"✅ **{barcode}** - {tipo_label}: {marca} {modelo} ({potencia}) - {processing_time:.1f}s")
        elif 'SCADA' in tipo_label:
            codigo = extraction_result.get('codigo_scada_principal', 'N/A')
            status_text.success(f"✅ **{barcode}** - {tipo_label}: Código {codigo} - {processing_time:.1f}s")
        else:
            status_text.success(f"✅ **{barcode}** - {tipo_label} procesado en {processing_time:.1f}s")
        
    except Exception as e:
        # Store error
        st.session_state.individual_results[row_key] = {
            'status': 'error',
            'barcode': barcode,
            'tipo_label': tipo_label,
            'error': str(e)
        }
        logger.error(f"Error processing {barcode} [{tipo_label}]: {e}")
        
        # Show error in UI
        status_text.error(f"❌ **{barcode}** - {tipo_label}: Error - {str(e)}")
    
    finally:
        # Remove from queue
        if row_key in st.session_state.processing_queue:
            st.session_state.processing_queue.remove(row_key)
        
        # Log queue status
        logger.info(f"Queue status after processing: {len(st.session_state.processing_queue)} items remaining")
        if st.session_state.processing_queue:
            logger.info(f"Next in queue: {st.session_state.processing_queue[0]}")
        
        # Save checkpoint after processing each item
        save_checkpoint_incremental()


def process_specific_batch(batch: dict, default_method: str, default_model: Optional[str]):
    """
    Process a specific batch of images with real-time monitoring
    
    Args:
        batch: Dictionary with batch info (batch_num, df, tipo_counts, etc.)
        default_method: Extraction method (API OpenAI, OCR Local, etc.)
        default_model: Model to use (gpt-4o-mini, etc.)
    """
    batch_df = batch['df']
    batch_num = batch['batch_num']
    total_images = len(batch_df)
    
    if total_images == 0:
        st.warning("⚠️ Este lote no tiene imágenes para procesar")
        return
    
    # Check if there are existing results to merge with
    existing_results = st.session_state.batch_results if st.session_state.batch_results else []
    if existing_results:
        st.info(f"ℹ️ Se encontraron {len(existing_results)} resultados previos. Los nuevos resultados se consolidarán con los existentes.")
    
    # Validation
    if default_method == "API OpenAI" and not os.getenv('OPENAI_API_KEY'):
        st.error("❌ API Key de OpenAI no configurada. Configúrala en la página de Transcripción Asistida.")
        return
    
    # Initialize API extractor if needed
    api_extractor = None
    if default_method == "API OpenAI":
        config_obj = st.session_state.batch_config
        api_config = config_obj.config.get('api', {}).copy() if hasattr(config_obj, 'config') else {}
        if default_model:
            api_config['model'] = default_model
        
        try:
            api_extractor = APIExtractor(api_config)
            logger.info(f"API Extractor initialized for batch {batch_num} with model: {api_config.get('model', 'gpt-4o-mini-2024-07-18')}")
            
            # TEST API CONNECTIVITY before processing
            st.info("🔍 Verificando conectividad con OpenAI API...")
            try:
                test_response = api_extractor.client.models.list()
                st.success(f"✅ Conexión con API exitosa - Modelos disponibles: {len(test_response.data)}")
                logger.info(f"API connectivity test passed - {len(test_response.data)} models available")
            except Exception as test_e:
                st.error(f"❌ Fallo en test de conectividad: {test_e}")
                st.warning("⚠️ No se puede continuar sin conexión a la API")
                logger.error(f"API connectivity test failed: {test_e}")
                return
            
            # VERIFY ALL IMAGES EXIST before starting
            st.info("📂 Verificando que todas las imágenes existen...")
            missing_images = []
            for _, row in batch_df.iterrows():
                img_path = Path(row['_img_path'])
                if not img_path.exists():
                    missing_images.append(str(img_path))
            
            if missing_images:
                st.error(f"❌ Faltan {len(missing_images)} imágenes")
                with st.expander("Ver imágenes faltantes"):
                    for img in missing_images[:20]:  # Show first 20
                        st.text(img)
                    if len(missing_images) > 20:
                        st.text(f"... y {len(missing_images) - 20} más")
                st.warning("⚠️ Verifica las rutas de las imágenes en el Excel")
                return
            else:
                st.success(f"✅ Todas las {total_images} imágenes encontradas")
            
        except Exception as e:
            st.error(f"❌ Error inicializando API: {e}")
            logger.error(f"API initialization error: {e}")
            return
    
    # Progress tracking with st.status (better for long operations)
    st.markdown("---")
    st.markdown(f"### 🚀 Procesando Lote {batch_num}")
    st.caption(f"📦 {total_images} imágenes | 🔖 Tipos: {', '.join([f'{k}({v})' for k, v in batch['tipo_counts'].items()])}")
    
    # Use st.status for better progress tracking
    status_container = st.status("Iniciando procesamiento...", expanded=True)
    
    with status_container:
        st.write(f"🎯 **Objetivo:** Procesar {total_images} imágenes")
        st.write(f"⏱️ **Tiempo estimado:** ~{total_images * 2.5 / 60:.0f} minutos")
        
        # Progress indicators
        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        
        # Log display
        log_display = st.empty()
    
    # Metrics display outside status
    metric_cols = st.columns(4)
    metric_barcode = metric_cols[0].empty()
    metric_progress = metric_cols[1].empty()
    metric_time = metric_cols[2].empty()
    metric_success = metric_cols[3].empty()
    
    # Results preview
    results_preview = st.empty()
    
    # Log buffer
    log_buffer = []
    successful_calls = 0
    failed_calls = 0
    
    # Track processing
    processed = 0
    results = []
    start_total_time = time.time()
    
    # Group images by BarCode
    barcode_groups = {}
    for _, row in batch_df.iterrows():
        barcode = row['BarCode']
        if barcode not in barcode_groups:
            barcode_groups[barcode] = []
        barcode_groups[barcode].append(row)
    
    # Helper function to add log entry
    def add_log(message: str, level: str = "info"):
        timestamp = datetime.now().strftime('%H:%M:%S')
        icon = "🔵" if level == "info" else "✅" if level == "success" else "❌" if level == "error" else "⚠️"
        log_buffer.append(f"`{timestamp}` {icon} {message}")
        # Keep only last 15 entries
        if len(log_buffer) > 15:
            log_buffer.pop(0)
        # Update display
        log_display.markdown("\n\n".join(log_buffer))
    
    # Helper to update metrics
    def update_metrics(barcode_val, progress_val, time_val, success_val):
        metric_barcode.metric("BarCode", barcode_val)
        metric_progress.metric("Progreso", progress_val)
        metric_time.metric("Tiempo restante", time_val)
        metric_success.metric("Éxito", success_val)
    
    add_log(f"Iniciando procesamiento de lote {batch_num} con {total_images} imágenes")
    update_metrics("Iniciando...", f"0/{total_images}", "Calculando...", "0%")
    
    # Process each barcode
    for barcode, rows in barcode_groups.items():
        # Update status
        update_metrics(barcode, f"{processed}/{total_images}", "Calculando...", f"{(successful_calls / max(1, successful_calls + failed_calls) * 100):.0f}%")
        st.session_state.batch_processing_status[barcode] = 'processing'
        add_log(f"Procesando BarCode: {barcode} ({len(rows)} imágenes)")
        
        barcode_results = {
            'BarCode': barcode,
            'images_processed': [],
            'processing_time': 0,
            'status': 'success',
            'method': default_method,
            'model': default_model,
            'results_by_type': {}
        }
        
        start_time = time.time()
        
        # Process each image of this barcode
        for row_idx, row in enumerate(rows, 1):
            # Update progress
            progress_pct = processed / total_images
            progress_bar.progress(progress_pct)
            progress_text.markdown(f"**{processed}/{total_images}** imágenes procesadas ({progress_pct*100:.1f}%)")
            
            elapsed = time.time() - start_total_time
            if processed > 0:
                avg_time = elapsed / processed
                remaining = avg_time * (total_images - processed)
                remaining_str = f"{remaining:.0f}s" if remaining < 120 else f"{remaining/60:.1f}min"
                
                # Success rate
                total_calls = successful_calls + failed_calls
                if total_calls > 0:
                    success_rate = (successful_calls / total_calls) * 100
                    update_metrics(barcode, f"{processed}/{total_images}", remaining_str, f"{success_rate:.0f}%")
            
            img_path = row['_img_path']
            tipo_label = row['Tipo']
            img_name = Path(img_path).name
            img_idx = row.get('_img_idx', '')
            
            # Generate row_key to check if already processed
            row_key = f"{barcode}_{tipo_label}_{img_idx}"
            
            # SKIP if already processed successfully
            if row_key in st.session_state.individual_results:
                existing_result = st.session_state.individual_results[row_key]
                if existing_result.get('status') == 'success':
                    logger.info(f"✓ Skipping {row_key} - already processed successfully")
                    add_log(f"{barcode} [{tipo_label}] ✓ Ya procesada (omitida)", "info")
                    
                    # Add to barcode_results from existing data
                    barcode_results['results_by_type'][tipo_label] = existing_result.get('data', {})
                    barcode_results['images_processed'].append({
                        'image_path': img_path,
                        'type': tipo_label,
                        'type_label': tipo_label,
                        'result': existing_result.get('data', {})
                    })
                    
                    processed += 1
                    successful_calls += 1
                    continue  # Skip API call
            
            try:
                # Map tipo_label to API image_type (handle auto-numbered suffixes)
                # Extract base type (remove _N suffix): PLACA1_2 -> PLACA1
                base_tipo = tipo_label.split('_')[0] if '_' in tipo_label else tipo_label
                
                tipo_mapping = {
                    'PLACA1': 'placa_tecnica',
                    'PLACA2': 'placa_tecnica',
                    'PLACA3': 'placa_tecnica',
                    'SCADA1': 'codigo_scada',
                    'SCADA2': 'codigo_scada',
                    'SCADA3': 'codigo_scada',
                    'AMBOS': None  # Auto-detect
                }
                image_type = tipo_mapping.get(base_tipo, None)
                
                logger.info(f"Type mapping: {tipo_label} (base: {base_tipo}) -> {image_type}")
                
                # Process image
                start_time = time.time()
                extraction_result = api_extractor.extract_from_image_assisted(
                    img_path,
                    barcode,
                    preprocessed_path=None,
                    image_type=image_type
                )
                processing_time = time.time() - start_time
                
                logger.info(f"API response received for {barcode} type {tipo_label} in {processing_time:.1f}s")
                
                # Store result by type_label
                barcode_results['results_by_type'][tipo_label] = {
                    'marca': extraction_result.get('marca'),
                    'modelo': extraction_result.get('modelo'),
                    'numero_serie': extraction_result.get('numero_serie'),
                    'año': extraction_result.get('año'),
                    'potencia': extraction_result.get('potencia'),
                    'codigo_scada_principal': extraction_result.get('codigo_scada_principal'),
                    'confidence': extraction_result.get('overall_confidence', 0.0),
                    'raw_response': extraction_result.get('raw_response')
                }
                
                barcode_results['images_processed'].append({
                    'image_path': img_path,
                    'type': image_type or 'auto',
                    'type_label': tipo_label,
                    'result': extraction_result
                })
                
                # SAVE to individual_results immediately
                st.session_state.individual_results[row_key] = {
                    'status': 'success',
                    'barcode': barcode,
                    'tipo_label': tipo_label,
                    'img_idx': str(img_idx),
                    'img_path': img_path,
                    'data': extraction_result,
                    'processing_time': processing_time
                }
                
                processed += 1
                successful_calls += 1
                add_log(f"{barcode} [{tipo_label}] ✓ OK ({processing_time:.1f}s)", "success")
                
                # INCREMENTAL SAVE: Save progress every 50 images (WITH individual_results)
                if processed % 50 == 0:
                    try:
                        temp_results = results if not existing_results else list({r['BarCode']: r for r in existing_results + results}.values())
                        output_dir = Path(st.session_state.batch_config.output_dir) / "batch_results"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        temp_file = output_dir / f"batch_results_lote{batch_num}_progress_{processed}.json"
                        
                        # Save in CHECKPOINT FORMAT (with individual_results for anti-reprocesamiento protection)
                        checkpoint_data = {
                            'batch_results': temp_results,
                            'individual_results': st.session_state.get('individual_results', {}),
                            'processing_queue': st.session_state.get('processing_queue', []),
                            'batch_selections': st.session_state.get('batch_selections', {}),
                            'timestamp': datetime.now().isoformat(),
                            'version': '2.0',
                            'metadata': {
                                'lote_num': batch_num,
                                'images_processed': processed,
                                'total_images': total_images,
                                'total_results': len(temp_results),
                                'total_individual': len(st.session_state.get('individual_results', {})),
                            }
                        }
                        
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
                        logger.info(f"Batch {batch_num} progress saved: {processed}/{total_images} images ({len(checkpoint_data['individual_results'])} individual)")
                        add_log(f"💾 Checkpoint guardado: {processed} imágenes", "success")
                    except Exception as e:
                        logger.error(f"Failed to save progress: {e}")
                        add_log(f"Error guardando checkpoint: {str(e)[:50]}", "error")
            
            except Exception as e:
                error_msg = str(e)[:100]
                logger.error(f"Error processing {barcode} image {img_path}: {e}")
                add_log(f"{barcode} [{tipo_label}] ✗ Error: {error_msg}", "error")
                failed_calls += 1
                
                barcode_results['status'] = 'error'
                barcode_results['error'] = str(e)
                st.session_state.batch_processing_status[barcode] = 'error'
                processed += 1
        
        barcode_results['processing_time'] = time.time() - start_time
        results.append(barcode_results)
        
        if barcode_results['status'] == 'success':
            st.session_state.batch_processing_status[barcode] = 'completed'
            add_log(f"✅ {barcode} completado - {len(rows)} imágenes procesadas", "success")
        
        # Show live preview
        with results_preview:
            summary_parts = [f"**Último:** {barcode}"]
            for type_label, data in barcode_results.get('results_by_type', {}).items():
                if 'PLACA' in type_label:
                    summary_parts.append(f"{type_label}: {data.get('marca', 'N/A')} {data.get('modelo', 'N/A')}")
                elif 'SCADA' in type_label:
                    summary_parts.append(f"{type_label}: {data.get('codigo_scada_principal', 'N/A')}")
            st.info(" | ".join(summary_parts))
    
    # Final update
    progress_bar.progress(1.0)
    progress_text.markdown(f"**✅ {processed}/{total_images}** imágenes completadas")
    elapsed_total = time.time() - start_total_time
    
    # Final success rate
    total_calls = successful_calls + failed_calls
    if total_calls > 0:
        success_rate = (successful_calls / total_calls) * 100
        update_metrics("✅ Completado", f"{processed}/{total_images}", f"{elapsed_total:.1f}s", f"{success_rate:.0f}%")
    
    add_log(f"🎉 Lote {batch_num} completado: {processed} imágenes en {elapsed_total:.1f}s", "success")
    
    # Update status to complete
    status_container.update(label=f"✅ Lote {batch_num} completado - {processed} imágenes procesadas", state="complete", expanded=False)
    
    # CONSOLIDATE: Merge with existing results
    if existing_results:
        existing_map = {r['BarCode']: r for r in existing_results}
        
        for new_result in results:
            barcode = new_result['BarCode']
            if barcode in existing_map:
                existing_map[barcode]['images_processed'].extend(new_result['images_processed'])
                existing_map[barcode]['results_by_type'].update(new_result['results_by_type'])
                existing_map[barcode]['processing_time'] += new_result['processing_time']
                logger.info(f"Merged results for BarCode: {barcode}")
            else:
                existing_map[barcode] = new_result
        
        consolidated_results = list(existing_map.values())
        st.session_state.batch_results = consolidated_results
        st.info(f"✅ Resultados consolidados: {len(consolidated_results)} activos totales")
    else:
        st.session_state.batch_results = results
    
    # AUTO-SAVE results (in CHECKPOINT FORMAT)
    try:
        output_dir = Path(st.session_state.batch_config.output_dir) / "batch_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = output_dir / f"batch_results_lote{batch_num}_{timestamp}.json"
        
        # Save in CHECKPOINT FORMAT (with individual_results for recovery)
        final_checkpoint = {
            'batch_results': st.session_state.batch_results,
            'individual_results': st.session_state.get('individual_results', {}),
            'processing_queue': st.session_state.get('processing_queue', []),
            'batch_selections': st.session_state.get('batch_selections', {}),
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',
            'metadata': {
                'lote_num': batch_num,
                'images_processed': processed,
                'total_images': total_images,
                'total_results': len(st.session_state.batch_results),
                'total_individual': len(st.session_state.get('individual_results', {})),
                'completed': True
            }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(final_checkpoint, f, indent=2, ensure_ascii=False, default=str)
        
        st.session_state.last_batch_results_file = str(json_file)
        logger.info(f"Batch {batch_num} results saved to: {json_file} ({len(final_checkpoint['individual_results'])} individual)")
    except Exception as e:
        logger.error(f"Failed to auto-save results: {e}")
    
    # Show completion
    total_activos = len(st.session_state.batch_results)
    st.success(f"✅ ¡Lote {batch_num} completado! {processed} imágenes procesadas. Total acumulado: {total_activos} activos en {elapsed_total:.1f}s")
    st.info("💡 **Resultados guardados y consolidados** - Procesa el siguiente lote o exporta los resultados", icon="💡")


def process_batch(default_method: str, default_model: Optional[str]):
    """Process all selected images in batch with real-time progress"""
    # Count total images to process
    total_images = sum(
        sum(1 for img in imgs.values() if img.get('process', False))
        for imgs in st.session_state.batch_selections.values()
    )
    
    if total_images == 0:
        st.warning("⚠️ No hay imágenes seleccionadas para procesar")
        return
    
    # Check if there are existing results to merge with
    existing_results = st.session_state.batch_results if st.session_state.batch_results else []
    if existing_results:
        st.info(f"ℹ️ Se encontraron {len(existing_results)} resultados previos. Los nuevos resultados se consolidarán con los existentes.")
    
    # Validation
    if default_method == "API OpenAI" and not os.getenv('OPENAI_API_KEY'):
        st.error("❌ API Key de OpenAI no configurada. Configúrala en la página de Transcripción Asistida.")
        return
    
    # Initialize API extractor if needed
    api_extractor = None
    if default_method == "API OpenAI":
        # Get API config from Config object properly
        config_obj = st.session_state.batch_config
        api_config = config_obj.config.get('api', {}).copy() if hasattr(config_obj, 'config') else {}
        if default_model:
            api_config['model'] = default_model
        
        try:
            api_extractor = APIExtractor(api_config)
            logger.info(f"API Extractor initialized with model: {api_config.get('model', 'gpt-4o-mini-2024-07-18')}")
        except Exception as e:
            st.error(f"❌ Error inicializando API: {e}")
            logger.error(f"API initialization error: {e}")
            return
    
    # Progress tracking with status container
    st.markdown("---")
    st.markdown("### 🚀 Procesamiento en Curso")
    
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0.0)
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            current_barcode_text = st.empty()
        with status_col2:
            progress_text = st.empty()
        with status_col3:
            time_text = st.empty()
    
    results_preview = st.empty()
    
    # Track processing
    processed = 0
    results = []
    start_total_time = time.time()
    
    # Process each barcode
    for barcode, images in st.session_state.batch_selections.items():
        images_to_process = [(idx, info) for idx, info in images.items() if info.get('process', False)]
        
        if not images_to_process:
            continue
        
        # Update status
        current_barcode_text.markdown(f"**Procesando:** `{barcode}`")
        st.session_state.batch_processing_status[barcode] = 'processing'
        
        barcode_results = {
            'BarCode': barcode,
            'images_processed': [],
            'processing_time': 0,
            'status': 'success',
            'method': default_method,
            'model': default_model,
            # Store results by type label (PLACA1, PLACA2, SCADA1, etc.)
            'results_by_type': {}
        }
        
        start_time = time.time()
        
        # Process each image
        for img_idx, img_info in images_to_process:
            # Update progress
            progress_pct = processed / total_images
            progress_bar.progress(progress_pct)
            progress_text.markdown(f"**Progreso:** {processed}/{total_images} imágenes")
            
            elapsed = time.time() - start_total_time
            if processed > 0:
                avg_time = elapsed / processed
                remaining = avg_time * (total_images - processed)
                time_text.markdown(f"**Tiempo restante:** ~{remaining:.0f}s")
            
            try:
                if default_method == "API OpenAI" and api_extractor:
                    # Log before API call
                    logger.info(f"Calling API for {barcode} image {img_idx}, type: {img_info.get('type_label', img_info['type'])}")
                    
                    # Extract using API
                    extraction_result = api_extractor.extract_from_image_assisted(
                        img_info['path'],
                        barcode,
                        preprocessed_path=None,
                        image_type=img_info['type']
                    )
                    
                    logger.info(f"API response received for {barcode} image {img_idx}")
                    
                    # Store result by type_label (PLACA1, PLACA2, SCADA1, etc.)
                    type_label = img_info.get('type_label', img_info['type'])
                    
                    barcode_results['results_by_type'][type_label] = {
                        'marca': extraction_result.get('marca'),
                        'modelo': extraction_result.get('modelo'),
                        'numero_serie': extraction_result.get('numero_serie'),
                        'año': extraction_result.get('año'),
                        'potencia': extraction_result.get('potencia'),
                        'codigo_scada_principal': extraction_result.get('codigo_scada_principal'),
                        'confidence': extraction_result.get('overall_confidence', 0.0),
                        'raw_response': extraction_result.get('raw_response')
                    }
                    
                    barcode_results['images_processed'].append({
                        'image_idx': img_idx,
                        'type': img_info['type'],
                        'type_label': type_label,
                        'result': extraction_result
                    })
                else:
                    # Other methods (OCR Local, Manual) - placeholder
                    barcode_results['images_processed'].append({
                        'image_idx': img_idx,
                        'type': img_info['type'],
                        'type_label': img_info.get('type_label', img_info['type']),
                        'result': {'error': 'Method not implemented yet'}
                    })
                
                processed += 1
                
                # INCREMENTAL SAVE: Use unified checkpoint system every 10 images
                if processed % 10 == 0:
                    try:
                        # Update batch_results with current progress before saving
                        temp_results = results if not existing_results else list({r['BarCode']: r for r in existing_results + results}.values())
                        st.session_state.batch_results = temp_results
                        
                        # Save using unified checkpoint system
                        save_checkpoint_incremental()
                        logger.info(f"✅ Checkpoint auto-saved: {processed}/{total_images} images")
                    except Exception as e:
                        logger.error(f"Failed to save checkpoint: {e}")
                
            except Exception as e:
                logger.error(f"Error processing {barcode} image {img_idx}: {e}")
                barcode_results['status'] = 'error'
                barcode_results['error'] = str(e)
                st.session_state.batch_processing_status[barcode] = 'error'
                processed += 1  # Count as processed even if error
        
        barcode_results['processing_time'] = time.time() - start_time
        results.append(barcode_results)
        
        if barcode_results['status'] == 'success':
            st.session_state.batch_processing_status[barcode] = 'completed'
        
        # Show live preview of current result (summarize all types)
        with results_preview:
            summary_parts = [f"✅ Completado: {barcode}"]
            for type_label, data in barcode_results.get('results_by_type', {}).items():
                if 'PLACA' in type_label:
                    summary_parts.append(f"{type_label}: {data.get('marca', 'N/A')} {data.get('modelo', 'N/A')}")
                elif 'SCADA' in type_label:
                    summary_parts.append(f"{type_label}: {data.get('codigo_scada_principal', 'N/A')}")
            st.caption(" | ".join(summary_parts))
    
    # Final update
    progress_bar.progress(1.0)
    current_barcode_text.markdown("**Estado:** ✅ Completado")
    progress_text.markdown(f"**Progreso:** {processed}/{total_images} imágenes")
    elapsed_total = time.time() - start_total_time
    time_text.markdown(f"**Tiempo total:** {elapsed_total:.1f}s")
    
    # CONSOLIDATE: Merge new results with existing results
    if existing_results:
        # Create a map of existing results by BarCode
        existing_map = {r['BarCode']: r for r in existing_results}
        
        # Merge new results
        for new_result in results:
            barcode = new_result['BarCode']
            if barcode in existing_map:
                # Update existing entry with new data
                existing_map[barcode]['images_processed'].extend(new_result['images_processed'])
                existing_map[barcode]['results_by_type'].update(new_result['results_by_type'])
                existing_map[barcode]['processing_time'] += new_result['processing_time']
                logger.info(f"Merged results for BarCode: {barcode}")
            else:
                # Add new entry
                existing_map[barcode] = new_result
        
        # Convert back to list
        consolidated_results = list(existing_map.values())
        st.session_state.batch_results = consolidated_results
        st.info(f"✅ Resultados consolidados: {len(consolidated_results)} activos totales")
    else:
        # No existing results, just use new results
        st.session_state.batch_results = results
    
    # AUTO-SAVE consolidated results to disk
    try:
        output_dir = Path(st.session_state.batch_config.output_dir) / "batch_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = output_dir / f"batch_results_{timestamp}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.batch_results, f, indent=2, ensure_ascii=False, default=str)
        
        st.session_state.last_batch_results_file = str(json_file)
        logger.info(f"Consolidated results auto-saved to: {json_file}")
    except Exception as e:
        logger.error(f"Failed to auto-save results: {e}")
    
    # Show completion message
    total_activos = len(st.session_state.batch_results)
    st.success(f"✅ ¡Procesamiento completado! {processed} imágenes nuevas procesadas. Total: {total_activos} activos en {elapsed_total:.1f}s")
    
    st.info("💡 **Resultados guardados automáticamente** - Ve a la pestaña **📊 Resultados** para ver la tabla completa y exportar a Excel", icon="💡")
    
    # Display results summary
    st.markdown("### 📊 Vista Previa de Resultados")
    
    for result in results:
        status_icon = '✅' if result['status'] == 'success' else '❌'
        with st.expander(f"{status_icon} **{result['BarCode']}** - {len(result['images_processed'])} imágenes ({result['processing_time']:.1f}s)"):
            if result['status'] == 'error':
                st.error(f"❌ Error: {result.get('error', 'Unknown')}")
            else:
                # Display results grouped by type (PLACA1, PLACA2, SCADA1, etc.)
                for type_label, data in result.get('results_by_type', {}).items():
                    st.markdown(f"#### {type_label}")
                    
                    if 'PLACA' in type_label:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"- **Marca:** {data.get('marca') or 'N/A'}")
                            st.write(f"- **Modelo:** {data.get('modelo') or 'N/A'}")
                            st.write(f"- **Potencia:** {data.get('potencia') or 'N/A'}")
                        with col2:
                            st.write(f"- **SN:** {data.get('numero_serie') or 'N/A'}")
                            st.write(f"- **Año:** {data.get('año') or 'N/A'}")
                            conf = data.get('confidence', 0.0)
                            st.markdown(get_confidence_badge(conf), unsafe_allow_html=True)
                    
                    elif 'SCADA' in type_label:
                        st.write(f"- **Código:** {data.get('codigo_scada_principal') or 'N/A'}")
                        conf = data.get('confidence', 0.0)
                        st.markdown(get_confidence_badge(conf), unsafe_allow_html=True)
                    
                    st.markdown("---")
                
                st.markdown(f"**⚙️ Método:** {result.get('method', 'N/A')}")
                if result.get('model'):
                    st.markdown(f"**🤖 Modelo:** {result['model']}")
    
    st.markdown("---")
    st.info("💡 **Tip:** Puedes exportar los resultados a Excel en la pestaña 'Resultados'")
    
    # Auto-switch to results tab would be nice, but not possible in Streamlit
    # User needs to manually click on Results tab


def save_configuration(selector_df: pd.DataFrame):
    """Save current selection configuration to disk"""
    # TEMPORARILY DISABLED to prevent infinite loop
    # This was causing 100+ saves per second blocking the UI
    return None
    
    output_dir = Path(st.session_state.batch_config.output_dir) / "batch_configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use excel filename as base for config name
    excel_name = Path(st.session_state.get('excel_path', 'unnamed')).stem
    config_file = output_dir / f"config_{excel_name}.json"
    
    # Save only relevant columns
    config_data = selector_df[['BarCode', 'Img', '✅', 'Tipo', '_img_idx', '_img_path', '_barcode_orig']].to_dict('records')
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    st.session_state.last_config_file = str(config_file)
    logger.info(f"Configuration saved to: {config_file}")
    return config_file


def load_configuration_from_file(file_path: Path) -> pd.DataFrame:
    """Load configuration from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        df = pd.DataFrame(config_data)
        logger.info(f"Configuration loaded from: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load configuration from {file_path}: {e}")
        return None


def find_configuration_file(excel_name: str) -> Optional[Path]:
    """Find configuration file for given Excel name"""
    output_dir = Path(st.session_state.batch_config.output_dir) / "batch_configs"
    
    if not output_dir.exists():
        return None
    
    config_file = output_dir / f"config_{excel_name}.json"
    
    if config_file.exists():
        return config_file
    
    return None


def load_previous_results():
    """Load previous batch results AND checkpoints from disk"""
    output_dir = Path(st.session_state.batch_config.output_dir) / "batch_results"
    
    if not output_dir.exists():
        return None
    
    # Find ALL result files: batch_results AND checkpoints
    batch_files = list(output_dir.glob("batch_results_*.json"))
    checkpoint_files = list(output_dir.glob("checkpoint_*.json"))
    
    # Combine and sort by modification time (most recent first)
    all_files = batch_files + checkpoint_files
    
    if not all_files:
        return None
    
    # Sort by modification time (newest first)
    json_files = sorted(all_files, key=lambda f: f.stat().st_mtime, reverse=True)
    
    return json_files


def reconstruct_individual_results_from_batch():
    """
    Reconstruct individual_results from batch_results when missing (for old checkpoints)
    This is critical for proper detection of already-processed images
    """
    if not st.session_state.get('batch_results'):
        logger.debug("No batch_results to reconstruct from")
        return 0
    
    if not st.session_state.get('individual_results'):
        st.session_state.individual_results = {}
    
    reconstructed_count = 0
    
    for batch_result in st.session_state.batch_results:
        barcode = batch_result['BarCode']
        results_by_type = batch_result.get('results_by_type', {})
        images_processed = batch_result.get('images_processed', [])
        
        # Method 1: Use images_processed list (most detailed)
        for img_info in images_processed:
            tipo_label = img_info.get('type_label', img_info.get('Tipo', 'PLACA1'))
            img_path = img_info.get('image_path', '')
            
            # Try to find matching img_idx from batch_selector_df
            img_idx = None
            if 'batch_selector_df' in st.session_state and st.session_state.batch_selector_df is not None:
                selector_df = st.session_state.batch_selector_df
                mask = (selector_df['BarCode'] == barcode) & \
                       (selector_df['Tipo'] == tipo_label) & \
                       (selector_df['_img_path'] == img_path)
                matching_rows = selector_df[mask]
                if not matching_rows.empty:
                    img_idx = matching_rows.iloc[0]['_img_idx']
            
            # Fallback: use tipo_label as identifier if no img_idx found
            if img_idx is None:
                # Generate a pseudo img_idx based on position in results_by_type
                tipo_keys = list(results_by_type.keys())
                img_idx = tipo_keys.index(tipo_label) if tipo_label in tipo_keys else 0
            
            row_key = f"{barcode}_{tipo_label}_{img_idx}"
            
            # Only reconstruct if not already present
            if row_key not in st.session_state.individual_results:
                st.session_state.individual_results[row_key] = {
                    'status': 'success',
                    'barcode': barcode,
                    'tipo_label': tipo_label,
                    'img_idx': str(img_idx),
                    'img_path': img_path,
                    'data': img_info.get('result', {}),
                    'reconstructed': True  # Mark as reconstructed for debugging
                }
                reconstructed_count += 1
                logger.debug(f"Reconstructed: {row_key}")
        
        # Method 2: Fallback to results_by_type if images_processed is empty
        if not images_processed and results_by_type:
            for tipo_label in results_by_type.keys():
                # Generate row_key with tipo_label (won't have img_idx)
                tipo_index = list(results_by_type.keys()).index(tipo_label)
                row_key = f"{barcode}_{tipo_label}_{tipo_index}"
                
                if row_key not in st.session_state.individual_results:
                    st.session_state.individual_results[row_key] = {
                        'status': 'success',
                        'barcode': barcode,
                        'tipo_label': tipo_label,
                        'img_idx': str(tipo_index),
                        'img_path': '',  # Unknown
                        'data': results_by_type[tipo_label],
                        'reconstructed': True
                    }
                    reconstructed_count += 1
                    logger.debug(f"Reconstructed (fallback): {row_key}")
    
    if reconstructed_count > 0:
        logger.info(f"✅ Reconstructed {reconstructed_count} individual_results entries from batch_results")
    
    return reconstructed_count


def reconstruct_batch_results_from_individual():
    """
    Reconstruct batch_results from individual_results (INVERSE operation)
    Used when checkpoint has individual_results but batch_results are missing or outdated
    Groups individual results by barcode and consolidates data
    """
    if not st.session_state.get('individual_results'):
        logger.debug("No individual_results to reconstruct from")
        return 0
    
    # Group individual_results by barcode
    results_by_barcode = {}
    
    for row_key, result_data in st.session_state.individual_results.items():
        if result_data.get('status') != 'success':
            continue  # Skip failed results
        
        barcode = result_data.get('barcode')
        if not barcode:
            # Try to extract from row_key (format: barcode_tipo_imgidx)
            parts = row_key.split('_')
            if len(parts) >= 2:
                barcode = parts[0]
            else:
                logger.warning(f"Cannot extract barcode from row_key: {row_key}")
                continue
        
        if barcode not in results_by_barcode:
            results_by_barcode[barcode] = {
                'BarCode': barcode,
                'results_by_type': {},
                'images_processed': [],
                'status': 'success',
                'processing_time': 0.0
            }
        
        # Add to results_by_type
        tipo_label = result_data.get('tipo_label', 'PLACA1')
        data = result_data.get('data', {})
        
        results_by_barcode[barcode]['results_by_type'][tipo_label] = data
        
        # Add to images_processed
        images_processed_entry = {
            'type_label': tipo_label,
            'image_path': result_data.get('img_path', ''),
            'result': data
        }
        results_by_barcode[barcode]['images_processed'].append(images_processed_entry)
        
        # Accumulate processing time if available
        if 'processing_time' in result_data:
            results_by_barcode[barcode]['processing_time'] += result_data.get('processing_time', 0.0)
    
    # Convert to list format
    new_batch_results = list(results_by_barcode.values())
    
    # Replace old batch_results
    st.session_state.batch_results = new_batch_results
    
    logger.info(f"✅ Reconstructed {len(new_batch_results)} batch_results from {len(st.session_state.individual_results)} individual_results")
    
    return len(new_batch_results)
    return reconstructed_count


def load_results_from_file(file_path: Path):
    """
    Load batch results from ANY valid JSON file and RESTORE FULL STATE
    Supports: checkpoint_*.json, batch_results_*.json, corrected_results_*.json
    Auto-detects format and extracts available data
    """
    try:
        # Try UTF-8 first
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            # Fallback to UTF-8 with error handling
            logger.warning(f"UTF-8 decode error, trying with errors='ignore'")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
        except json.JSONDecodeError as je:
            logger.error(f"JSON decode error at line {je.lineno}, col {je.colno}: {je.msg}")
            raise ValueError(f"Archivo JSON corrupto o inválido: {je.msg}")
        
        logger.info(f"📂 Loading from {file_path.name} ({file_path.stat().st_size / 1024:.1f}KB)")
        
        # === AUTO-DETECT FILE FORMAT ===
        file_type = "unknown"
        
        # 1. Checkpoint format (full state)
        if isinstance(data, dict) and 'batch_results' in data:
            file_type = "checkpoint"
            checkpoint_data = data
            results = checkpoint_data.get('batch_results', [])
            individual_results = checkpoint_data.get('individual_results', {})
            batch_selections = checkpoint_data.get('batch_selections', {})
            processing_queue = checkpoint_data.get('processing_queue', [])
            metadata = checkpoint_data.get('metadata', {})
            
        # 2. Legacy batch_results format (array of results)
        elif isinstance(data, list) and len(data) > 0 and 'BarCode' in data[0]:
            file_type = "batch_results"
            results = data
            individual_results = {}
            batch_selections = {}
            processing_queue = []
            metadata = {}
            
        # 3. Wrapped results format (dict with results key)
        elif isinstance(data, dict) and len(data) == 1:
            # Try to find results in any key
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    file_type = "wrapped_results"
                    results = value
                    individual_results = {}
                    batch_selections = {}
                    processing_queue = []
                    metadata = {}
                    break
            else:
                raise ValueError("No valid results found in file")
                
        # 4. Direct dict format (single result object)
        elif isinstance(data, dict) and 'BarCode' in data:
            file_type = "single_result"
            results = [data]  # Wrap in array
            individual_results = {}
            batch_selections = {}
            processing_queue = []
            metadata = {}
            
        else:
            raise ValueError(f"Unknown file format. Expected checkpoint, batch_results, or wrapped format.")
        
        logger.info(f"📋 Detected format: {file_type}")
        logger.info(f"📊 Content: {len(results)} results, {len(individual_results)} individual, {len(processing_queue)} queued")
        
        # Restore batch_results
        st.session_state.batch_results = results
        st.session_state.last_batch_results_file = file_path
        
        # Restore individual_results
        st.session_state.individual_results = individual_results
        
        # Restore processing queue
        st.session_state.processing_queue = processing_queue
        
        # Restore batch_selections (image paths by barcode)
        st.session_state.batch_selections = batch_selections
        
        # === RECONSTRUCT individual_results if missing (for old checkpoints) ===
        # DISABLED: Reconstruction is too slow for large datasets (takes ~20s per activo)
        # Only reconstruct if user explicitly needs anti-reprocesamiento protection
        if not individual_results or len(individual_results) == 0:
            logger.warning("⚠️ individual_results is empty (skipping reconstruction for performance)")
            logger.info("💡 To enable anti-reprocesamiento, load a checkpoint with individual_results included")
            # reconstructed = reconstruct_individual_results_from_batch()
            # logger.info(f"🔧 Reconstructed {reconstructed} individual_results entries")
        
        # === RECONSTRUCT batch_selector_df from batch_results ===
        # This is CRITICAL for showing images in Selección tab
        if st.session_state.batch_results and st.session_state.batch_data_df is not None:
            logger.info("🔄 Reconstructing batch_selector_df from loaded results...")
            
            # Get original DataFrame structure
            original_df = st.session_state.batch_data_df.copy()
            
            # Create selector_df from original if not exists
            if 'batch_selector_df' not in st.session_state or st.session_state.batch_selector_df is None:
                st.session_state.batch_selector_df = original_df.copy()
                if '✅' not in st.session_state.batch_selector_df.columns:
                    st.session_state.batch_selector_df['✅'] = False
            
            # Mark processed images as checked and update their types
            selector_df = st.session_state.batch_selector_df
            
            for result in results:
                barcode = result['BarCode']
                results_by_type = result.get('results_by_type', {})
                images_processed = result.get('images_processed', [])
                
                # Match barcode rows in selector_df
                barcode_mask = selector_df['BarCode'] == barcode
                barcode_rows = selector_df[barcode_mask]
                
                # Mark images as checked based on processed types
                for tipo_label in results_by_type.keys():
                    # Find matching rows with this tipo
                    tipo_mask = barcode_mask & (selector_df['Tipo'] == tipo_label)
                    if tipo_mask.any():
                        selector_df.loc[tipo_mask, '✅'] = True
                        logger.debug(f"  ✓ Marked {barcode} [{tipo_label}] as checked")
                
                # Also mark based on images_processed list
                for img_info in images_processed:
                    img_tipo = img_info.get('type_label', img_info.get('Tipo', 'PLACA1'))
                    img_idx = img_info.get('image_idx', img_info.get('Img', 0))
                    
                    # Find exact match by barcode, tipo, and img index
                    exact_mask = (selector_df['BarCode'] == barcode) & \
                                (selector_df['Tipo'] == img_tipo) & \
                                (selector_df['Img'] == img_idx)
                    
                    if exact_mask.any():
                        selector_df.loc[exact_mask, '✅'] = True
            
            st.session_state.batch_selector_df = selector_df
            logger.info(f"✅ batch_selector_df reconstructed with {len(results)} barcodes")
        
        # Show success message with details
        total_images = sum(len(r.get('images_processed', [])) for r in results)
        unique_types = set()
        for r in results:
            unique_types.update(r.get('results_by_type', {}).keys())
        
        st.success(f"✅ Resultados cargados: {len(results)} activos, {len(individual_results)} imágenes procesadas")
        
        if individual_results:
            # Show distribution by type
            type_counts = {}
            for img_key, img_data in individual_results.items():
                tipo = img_data.get('tipo_label', 'Unknown')
                type_counts[tipo] = type_counts.get(tipo, 0) + 1
            
            type_summary = ", ".join([f"{k}: {v}" for k, v in sorted(type_counts.items())])
            st.info(f"📊 Distribución por tipo: {type_summary}")
        
        if individual_results:
            st.info(f"📊 Estado restaurado: {len(individual_results)} resultados individuales")
        
        if processing_queue:
            st.warning(f"⏳ Cola de procesamiento: {len(processing_queue)} imágenes pendientes")
        
        logger.info(f"✅ FULL STATE LOADED:")
        logger.info(f"   - {len(results)} batch_results")
        logger.info(f"   - {len(individual_results)} individual_results")
        logger.info(f"   - {len(batch_selections)} batch_selections")
        logger.info(f"   - {len(processing_queue)} processing_queue")
        logger.info(f"   - Types found: {unique_types}")
        
        # === UPDATE CHECKPOINT BANNER ===
        # Update the checkpoint_restored state to show correct info at top of page
        file_timestamp = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        
        st.session_state.checkpoint_restored = {
            'queue_len': len(processing_queue),
            'results_len': len(individual_results) if individual_results else total_images,
            'timestamp': file_timestamp,
            'source_file': file_path.name
        }
        
        logger.info(f"✅ Checkpoint banner updated: {len(individual_results)} images, {len(processing_queue)} queued, from {file_path.name}")
        
        # === SAVE AS INCREMENTAL CHECKPOINT ===
        # This ensures the loaded state persists across reloads/reruns
        # Otherwise the old checkpoint_incremental_latest.json will overwrite on next rerun
        try:
            save_checkpoint_incremental()
            logger.info(f"✅ Loaded checkpoint saved as new incremental checkpoint (will persist across reloads)")
        except Exception as save_error:
            logger.warning(f"Could not save as incremental checkpoint: {save_error}")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error cargando resultados: {e}")
        logger.error(f"Failed to load results from {file_path}: {e}", exc_info=True)
        return False


def save_checkpoint_incremental():
    """
    Save COMPLETE checkpoint after each processed image in incremental mode
    Includes: batch_results, individual_results, batch_selections, processing_queue, and selector_df state
    """
    try:
        if not hasattr(st.session_state, 'batch_config'):
            return
        
        # Allow user to specify save directory
        if 'checkpoint_save_directory' in st.session_state and st.session_state.checkpoint_save_directory:
            output_dir = Path(st.session_state.checkpoint_save_directory)
        else:
            output_dir = Path(st.session_state.batch_config.output_dir) / "batch_results"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = output_dir / "checkpoint_incremental_latest.json"
        
        # Also save with timestamp for backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_checkpoint = output_dir / f"checkpoint_{timestamp}.json"
        
        # Prepare COMPLETE checkpoint data
        checkpoint_data = {
            'batch_results': st.session_state.get('batch_results', []),
            'processing_queue': st.session_state.get('processing_queue', []),
            'individual_results': st.session_state.get('individual_results', {}),
            'batch_selections': st.session_state.get('batch_selections', {}),
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',  # Updated version with full state
            'metadata': {
                'total_results': len(st.session_state.get('batch_results', [])),
                'total_individual': len(st.session_state.get('individual_results', {})),
                'queue_length': len(st.session_state.get('processing_queue', [])),
            }
        }
        
        # Save to latest checkpoint (always overwrite)
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Checkpoint saved: {len(checkpoint_data['batch_results'])} results, {len(checkpoint_data['processing_queue'])} in queue")
        
        # Save backup checkpoint every 10 processed items
        num_processed = len(checkpoint_data['batch_results'])
        if num_processed > 0 and num_processed % 10 == 0:
            with open(backup_checkpoint, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"💾 Backup checkpoint saved: {backup_checkpoint.name}")
        
        # === SYNC TO SHARED RESULTS ===
        try:
            shared_manager = SharedResultsManager(st.session_state.batch_config.output_dir)
            imported = shared_manager.import_from_procesamiento_rapido(checkpoint_data['batch_results'])
            logger.info(f"✅ Synced {imported} results to shared storage")
        except Exception as sync_error:
            logger.warning(f"Failed to sync to shared results: {sync_error}")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}", exc_info=True)


def restore_checkpoint_incremental() -> bool:
    """Restore checkpoint on app startup if available"""
    try:
        if not hasattr(st.session_state, 'batch_config'):
            logger.debug("No batch_config available, skipping checkpoint restore")
            return False
        
        output_dir = Path(st.session_state.batch_config.output_dir) / "batch_results"
        checkpoint_file = output_dir / "checkpoint_incremental_latest.json"
        
        if not checkpoint_file.exists():
            logger.debug(f"No checkpoint file found at {checkpoint_file}")
            return False
        
        logger.info(f"📂 Loading checkpoint from {checkpoint_file}...")
        
        # Load checkpoint
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        # Restore state
        st.session_state.batch_results = checkpoint_data.get('batch_results', [])
        st.session_state.processing_queue = checkpoint_data.get('processing_queue', [])
        st.session_state.individual_results = checkpoint_data.get('individual_results', {})
        st.session_state.batch_selections = checkpoint_data.get('batch_selections', {})
        
        timestamp = checkpoint_data.get('timestamp', 'unknown')
        queue_len = len(st.session_state.processing_queue)
        results_len = len(st.session_state.batch_results)
        
        logger.info(f"✅ Checkpoint restored from {timestamp}:")
        logger.info(f"   - {results_len} batch_results")
        logger.info(f"   - {queue_len} items in processing_queue")
        logger.info(f"   - {len(st.session_state.individual_results)} individual_results")
        logger.info(f"   - Queue contents: {st.session_state.processing_queue}")
        
        # Show notification to user
        if queue_len > 0 or results_len > 0:
            st.session_state.checkpoint_restored = {
                'queue_len': queue_len,
                'results_len': results_len,
                'timestamp': timestamp
            }
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to restore checkpoint: {e}")
        return False


def clear_checkpoint_incremental():
    """Clear checkpoint file and session state"""
    try:
        if not hasattr(st.session_state, 'batch_config'):
            return
        
        output_dir = Path(st.session_state.batch_config.output_dir) / "batch_results"
        checkpoint_file = output_dir / "checkpoint_incremental_latest.json"
        
        # Delete checkpoint file
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("Checkpoint file deleted")
        
        # Clear session state
        st.session_state.batch_results = []
        st.session_state.processing_queue = []
        st.session_state.individual_results = {}
        st.session_state.checkpoint_restored = None
        
        st.success("✅ Checkpoint limpiado - Comenzando desde cero")
        logger.info("Checkpoint cleared from session state")
        
    except Exception as e:
        st.error(f"❌ Error limpiando checkpoint: {e}")
        logger.error(f"Failed to clear checkpoint: {e}")


def export_results():
    """Export batch results to Excel with intelligent merging by BarCode and pair number"""
    if not st.session_state.batch_results:
        st.warning("No hay resultados para exportar")
        return
    
    # Group by BarCode and merge PLACA/SCADA pairs
    merged_data = {}
    
    for result in st.session_state.batch_results:
        barcode = result['BarCode']
        
        if barcode not in merged_data:
            merged_data[barcode] = {}
        
        # Process each type (PLACA1, SCADA1, PLACA2, SCADA2, etc.)
        for type_label, data in result.get('results_by_type', {}).items():
            # Extract pair number (1, 2, 3, etc.) from type label
            pair_num = '1'  # Default
            if 'PLACA' in type_label or 'SCADA' in type_label:
                # Extract number from PLACA1, PLACA2, SCADA1, SCADA2, etc.
                import re
                match = re.search(r'(\d+)', type_label)
                if match:
                    pair_num = match.group(1)
            
            if pair_num not in merged_data[barcode]:
                merged_data[barcode][pair_num] = {
                    'placa_data': {},
                    'scada_data': {}
                }
            
            # Store in appropriate category
            if 'PLACA' in type_label or type_label == 'AMBOS':
                merged_data[barcode][pair_num]['placa_data'] = {
                    'marca': data.get('marca', ''),
                    'modelo': data.get('modelo', ''),
                    'potencia': data.get('potencia', ''),
                    'numero_serie': data.get('numero_serie', ''),
                    'año': data.get('año', '')
                }
            
            if 'SCADA' in type_label or type_label == 'AMBOS':
                merged_data[barcode][pair_num]['scada_data'] = {
                    'codigo_scada': data.get('codigo_scada_principal', '')
                }
    
    # Convert to flat export format
    export_data = []
    for barcode, pairs in merged_data.items():
        for pair_num, pair_data in pairs.items():
            placa = pair_data['placa_data']
            scada = pair_data['scada_data']
            
            # Merge PLACA and SCADA data into single row
            row = {
                'BarCode': barcode,
                'Par': pair_num,  # Indicates which PLACA/SCADA pair (1, 2, 3, etc.)
                'Marca': placa.get('marca', ''),
                'Modelo': placa.get('modelo', ''),
                'Potencia': placa.get('potencia', ''),
                'Número de Serie': placa.get('numero_serie', ''),
                'Año': placa.get('año', ''),
                'Código SCADA': scada.get('codigo_scada', '')
            }
            
            # Only add row if it has some data
            if any([row['Marca'], row['Modelo'], row['Número de Serie'], row['Año'], row['Código SCADA'], row['Potencia']]):
                export_data.append(row)
    
    df_export = pd.DataFrame(export_data)
    
    # Sort by BarCode and Par for better readability
    df_export = df_export.sort_values(['BarCode', 'Par'])
    
    # Prepare Excel file in memory for download
    from io import BytesIO
    buffer = BytesIO()
    df_export.to_excel(buffer, index=False, engine='openpyxl')
    excel_data = buffer.getvalue()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"resultados_procesamiento_{timestamp}.xlsx"
    
    # Store in session state for persistent download button
    st.session_state.excel_export_data = excel_data
    st.session_state.excel_export_filename = filename
    st.session_state.excel_export_df = df_export
    st.session_state.excel_export_stats = {
        'rows': len(df_export),
        'barcodes': len(merged_data)
    }
    
    # Also save backup copy locally
    output_dir = Path(st.session_state.batch_config.output_dir) / "batch_results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / filename
    df_export.to_excel(output_file, index=False)
    
    # Also save JSON with full details
    json_file = output_dir / f"batch_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.batch_results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"💾 Excel backup saved: {output_file}")
    
    return output_file


def get_confidence_badge(confidence: float) -> str:
    """Get HTML badge for confidence level"""
    if confidence >= 0.8:
        color = "#28a745"
        label = "Alta"
    elif confidence >= 0.5:
        color = "#ffc107"
        label = "Media"
    else:
        color = "#dc3545"
        label = "Baja"
    
    return f'<span style="color: {color}; font-weight: bold;">✓ {label}</span>'


def display_review_interface():
    """Display review/edit interface with TRUE on-demand loading (only current activo)"""
    
    # Section to load previous results
    st.markdown("### 📂 Cargar Resultados Previos")
    
    previous_files = load_previous_results()
    
    if previous_files:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create dropdown with file names
            file_options = {f.name: f for f in previous_files}
            selected_file_name = st.selectbox(
                "Selecciona un archivo de resultados previos:",
                options=list(file_options.keys()),
                key="load_results_selector"
            )
        
        with col2:
            if st.button("📥 Cargar", use_container_width=True, key="load_results_btn"):
                selected_file = file_options[selected_file_name]
                load_results_from_file(selected_file)
                st.rerun()
    else:
        st.info("No hay resultados previos guardados. Procesa algunas imágenes primero.")
    
    st.markdown("---")
    
    # Review interface
    if not st.session_state.batch_results:
        st.info("No hay resultados cargados. Carga resultados previos o procesa nuevas imágenes.")
        return
    
    st.markdown("### 🔍 Revisión y Edición de Resultados")
    
    # === TRUE ON-DEMAND: Create lightweight index, load full activo only when needed ===
    # Build lightweight index (ONLY BarCode + index) if not exists
    if 'batch_results_index' not in st.session_state or st.session_state.batch_results_index is None:
        logger.info("🏗️ Building lightweight index from batch_results...")
        st.session_state.batch_results_index = [
            {'idx': i, 'BarCode': result.get('BarCode', f'Unknown_{i}')} 
            for i, result in enumerate(st.session_state.batch_results)
        ]
        logger.info(f"✅ Index built: {len(st.session_state.batch_results_index)} entries (lightweight)")
    
    # Cache for currently loaded activo (ONLY ONE at a time)
    if 'current_activo_cache' not in st.session_state:
        st.session_state.current_activo_cache = None
    if 'current_activo_cache_idx' not in st.session_state:
        st.session_state.current_activo_cache_idx = None
    
    total_results = len(st.session_state.batch_results_index)
    
    
    # Initialize review index if not exists
    if 'review_index' not in st.session_state:
        # Try to restore from backup first
        if '_review_index_backup' in st.session_state:
            st.session_state.review_index = st.session_state['_review_index_backup']
            logger.info(f"🔄 Restored review_index from backup: {st.session_state.review_index}")
        else:
            st.session_state.review_index = 0
    
    # Clamp review_index to valid range
    if st.session_state.review_index >= total_results:
        st.session_state.review_index = total_results - 1
    if st.session_state.review_index < 0:
        st.session_state.review_index = 0
    
    current_idx = st.session_state.review_index
    
    # === ON-DEMAND LOADING: Load ONLY current activo if not in cache ===
    if st.session_state.current_activo_cache_idx != current_idx:
        # Clear previous activo from cache
        if st.session_state.current_activo_cache_idx is not None:
            logger.info(f"🧹 Cleared activo #{st.session_state.current_activo_cache_idx + 1} from cache")
        
        # Load current activo from batch_results
        logger.info(f"📥 Loading activo #{current_idx + 1} on-demand...")
        st.session_state.current_activo_cache = st.session_state.batch_results[current_idx]
        st.session_state.current_activo_cache_idx = current_idx
        logger.info(f"✅ Activo #{current_idx + 1} loaded: {st.session_state.current_activo_cache.get('BarCode', 'N/A')}")
    
    # Get current activo from cache (only ONE in memory at a time)
    current_result = st.session_state.current_activo_cache
    
    # Show memory-friendly message for large datasets
    if total_results > 100:
        st.info(f"💾 **Modo optimizado activado**: Solo el activo actual está cargado en memoria (1 de {total_results})")
    
    # Navigation - Simple prev/next (no pagination needed with on-demand loading)
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])
    
    with nav_col1:
        if st.button("⏮️ Primero", disabled=(current_idx == 0), use_container_width=True, help="Ir al primer activo"):
            st.session_state.review_index = 0
            st.rerun()
    
    with nav_col2:
        if st.button("⬅️ Anterior", disabled=(current_idx == 0), use_container_width=True):
            st.session_state.review_index = max(0, current_idx - 1)
            st.rerun()
    
    with nav_col3:
        st.markdown(f"<div style='text-align: center; font-size: 1.2em; font-weight: bold;'>Activo {current_idx + 1} de {total_results}: {current_result['BarCode']}</div>", unsafe_allow_html=True)
    
    with nav_col4:
        if st.button("Siguiente ➡️", disabled=(current_idx >= total_results - 1), use_container_width=True):
            st.session_state.review_index = min(total_results - 1, current_idx + 1)
            st.rerun()
    
    with nav_col5:
        if st.button("⏭️ Último", disabled=(current_idx >= total_results - 1), use_container_width=True, help="Ir al último activo"):
            st.session_state.review_index = total_results - 1
            st.rerun()
    
    # Direct jump input
    col_jump1, col_jump2 = st.columns([2, 1])
    with col_jump1:
        jump_to = st.number_input(
            "Ir al activo #",
            min_value=1,
            max_value=total_results,
            value=current_idx + 1,
            step=1,
            key="jump_to_input",
            help="Ingresa el número de activo y presiona Enter"
        )
        if jump_to - 1 != current_idx:
            st.session_state.review_index = jump_to - 1
            st.rerun()
    with col_jump2:
        st.metric("Total", f"{total_results} activos")
    
    # === SEARCH/FILTER BAR FOR QUICK ACCESS ===
    st.markdown("---")
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_barcode = st.text_input(
            "🔍 Buscar por BarCode:",
            placeholder="Ingresa el BarCode para saltar directamente",
            key="review_search_barcode",
            help="Escribe el BarCode y presiona Enter o el botón Buscar"
        )
    
    with search_col2:
        if st.button("🔍 Buscar", use_container_width=True, key="search_barcode_btn"):
            if search_barcode:
                # Find the index of the barcode using lightweight index
                for idx, index_entry in enumerate(st.session_state.batch_results_index):
                    if index_entry['BarCode'].upper() == search_barcode.upper():
                        # Jump directly to that activo
                        st.session_state.review_index = idx
                        st.success(f"✅ Encontrado: {index_entry['BarCode']} (Activo #{idx + 1})")
                        st.rerun()
                        break
                else:
                    st.error(f"❌ BarCode '{search_barcode}' no encontrado en los resultados")
    
    st.markdown("---")
    
    # Display images for this barcode - LAZY LOADING ENABLED
    images_col, form_col = st.columns([1, 2])
    
    with images_col:
        st.markdown("#### 📸 Imágenes Procesadas")
        
        # Show fusion info if multiple images with same type
        images_by_type = {}
        for img_info in current_result.get('images_processed', []):
            tipo = img_info.get('type_label', 'unknown')
            if tipo not in images_by_type:
                images_by_type[tipo] = []
            images_by_type[tipo].append(img_info)
        
        # Check if any type has multiple images (fusion occurred)
        fusion_detected = any(len(imgs) > 1 for imgs in images_by_type.values())
        
        if fusion_detected:
            # Get the types that were fused
            fused_types = [tipo for tipo, imgs in images_by_type.items() if len(imgs) > 1]
            st.info(f"🔀 **Fusión Inteligente**: Las imágenes con el mismo sufijo ({', '.join(fused_types)}) se fusionaron automáticamente. Los datos se combinaron priorizando valores no-nulos.")
            st.caption("💡 Tip: Si son equipos distintos, procésalas con diferentes sufijos (ej: PLACA1_1, PLACA1_2).")
        
        images_found = False
        
        # Initialize image transformation states if not exists
        if 'image_rotation' not in st.session_state:
            st.session_state.image_rotation = {}
        if 'image_zoom' not in st.session_state:
            st.session_state.image_zoom = {}
        
        # ON-DEMAND LOADING: Load ONLY current activo, clear previous from memory
        if 'loaded_activo_images' not in st.session_state:
            st.session_state.loaded_activo_images = {}  # Cache for current activo only
        if 'last_loaded_activo_idx' not in st.session_state:
            st.session_state.last_loaded_activo_idx = None
        
        activo_key = f"activo_{current_idx}_{current_result['BarCode']}"
        
        # Clear previous activo images from memory if we navigated to a different activo
        if st.session_state.last_loaded_activo_idx != current_idx:
            st.session_state.loaded_activo_images.clear()  # Free memory
            st.session_state.last_loaded_activo_idx = current_idx
            logger.info(f"🧹 Cleared previous activo images from memory. Now on activo {current_idx + 1}")
        
        # Show images list first WITHOUT loading
        st.caption(f"📋 **{len(current_result.get('images_processed', []))} imagen(es) disponibles**")
        
        # Initialize checkbox state if not exists for this activo
        checkbox_key = f"toggle_images_{activo_key}"
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = False
        
        # Callback function to preserve index when checkbox changes
        def preserve_index_on_checkbox_change():
            """Preserves the review_index before Streamlit reruns"""
            st.session_state.review_index = current_idx
            st.session_state['_review_index_backup'] = current_idx
            logger.info(f"📌 Checkbox callback: Preserved review_index={current_idx}")
        
        # Toggle to load images ON DEMAND with callback
        show_images = st.checkbox(
            "🖼️ Cargar y mostrar imágenes",
            value=st.session_state[checkbox_key],
            key=checkbox_key,
            help="Activa para cargar las imágenes SOLO de este activo (libera memoria del anterior)",
            on_change=preserve_index_on_checkbox_change
        )
        
        if not show_images:
            # Just show metadata without loading images
            st.info("👆 Activa la casilla arriba para ver las imágenes")
            for idx, img_info in enumerate(current_result.get('images_processed', [])):
                img_idx = img_info.get('image_idx', img_info.get('Img', 0))
                img_type = img_info.get('type_label', img_info.get('type', 'unknown'))
                st.caption(f"• Imagen #{img_idx + 1} - {img_type}")
        else:
            # Load and display images only when requested
            # Track displayed images to avoid duplicates
            displayed_images = set()
            img_counter = 0
            
            for img_info in current_result.get('images_processed', []):
                img_idx = img_info.get('image_idx', img_info.get('Img', 0))
                img_type = img_info.get('type_label', img_info.get('type', img_info.get('Tipo', 'unknown')))
                
                # Try multiple sources for image path (prioritize direct paths first)
                img_path = None
                
                # 1. Try direct path in img_info (queue processing saves here)
                img_path = img_info.get('image_path') or img_info.get('path')
                
                # 2. Try from result data (old format compatibility)
                if not img_path and 'result' in img_info:
                    img_path = img_info['result'].get('image_path') or img_info['result'].get('img_path')
                
                # 3. Try from batch_selections as last resort
                if not img_path:
                    barcode = current_result['BarCode']
                    if barcode in st.session_state.get('batch_selections', {}):
                        if img_idx in st.session_state.batch_selections[barcode]:
                            img_path = st.session_state.batch_selections[barcode][img_idx].get('path')
                
                # Skip if already displayed (prevents duplicate keys)
                if img_path and img_path in displayed_images:
                    continue
                
                # Display image if path found
                if img_path and Path(img_path).exists():
                    try:
                        # Mark as displayed
                        displayed_images.add(img_path)
                        
                        # Create unique key for this image using counter and review index
                        img_key = f"review_{current_idx}_{img_counter}_{current_result['BarCode']}_{img_idx}"
                        img_counter += 1
                        
                        # Initialize transformation values for this image
                        if img_key not in st.session_state.image_rotation:
                            st.session_state.image_rotation[img_key] = 0
                        if img_key not in st.session_state.image_zoom:
                            st.session_state.image_zoom[img_key] = 100
                        
                        # Load image
                        img = Image.open(img_path)
                        
                        # Image controls
                        st.markdown(f"**#{img_idx + 1} - {img_type}**")
                        
                        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 1])
                        
                        with ctrl_col1:
                            if st.button("↶", key=f"rotate_left_{img_key}", help="Rotar 90° izquierda"):
                                st.session_state.image_rotation[img_key] = (st.session_state.image_rotation[img_key] - 90) % 360
                                st.rerun()
                        
                        with ctrl_col2:
                            if st.button("↷", key=f"rotate_right_{img_key}", help="Rotar 90° derecha"):
                                st.session_state.image_rotation[img_key] = (st.session_state.image_rotation[img_key] + 90) % 360
                                st.rerun()
                        
                        with ctrl_col3:
                            if st.button("🔍+", key=f"zoom_in_{img_key}", help="Aumentar zoom"):
                                st.session_state.image_zoom[img_key] = min(st.session_state.image_zoom[img_key] + 25, 200)
                                st.rerun()
                        
                        with ctrl_col4:
                            if st.button("🔍-", key=f"zoom_out_{img_key}", help="Reducir zoom"):
                                st.session_state.image_zoom[img_key] = max(st.session_state.image_zoom[img_key] - 25, 50)
                                st.rerun()
                        
                        # Display zoom level
                        st.caption(f"🔄 Rotación: {st.session_state.image_rotation[img_key]}° | Zoom: {st.session_state.image_zoom[img_key]}%")
                        
                        # Apply transformations
                        rotation = st.session_state.image_rotation[img_key]
                        zoom = st.session_state.image_zoom[img_key]
                        
                        # Rotate image if needed
                        if rotation != 0:
                            img = img.rotate(-rotation, expand=True)  # PIL rotates counter-clockwise
                        
                        # Apply zoom (resize)
                        original_size = img.size
                        zoom_factor = zoom / 100
                        new_size = (int(original_size[0] * zoom_factor), int(original_size[1] * zoom_factor))
                        
                        # Limit max size to prevent memory issues
                        max_dimension = 1200
                        if max(new_size) > max_dimension:
                            scale = max_dimension / max(new_size)
                            new_size = (int(new_size[0] * scale), int(new_size[1] * scale))
                        
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                        
                        # Display image with scrollable container if zoomed
                        if zoom > 100:
                            st.markdown(
                                f"""
                                <div style="max-height: 400px; overflow: auto; border: 1px solid #ddd; padding: 5px;">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        st.image(img, use_container_width=(zoom <= 100))
                        
                        # Reset button
                        if st.button("🔄 Resetear", key=f"reset_{img_key}", help="Volver a vista original"):
                            st.session_state.image_rotation[img_key] = 0
                            st.session_state.image_zoom[img_key] = 100
                            st.rerun()
                        
                        st.markdown("---")
                        images_found = True
                    except Exception as e:
                        st.error(f"Error cargando imagen #{img_idx + 1}: {e}")
                else:
                    st.caption(f"⚠️ Imagen #{img_idx + 1} - {img_type} (ruta no disponible)")
        
        if not images_found and not current_result.get('images_processed'):
            st.info("No hay imágenes asociadas a este activo")
        
        # Show raw API response if available (with safe JSON parsing)
        if current_result.get('images_processed'):
            with st.expander("📄 Ver respuestas API"):
                import json
                for idx, img_info in enumerate(current_result['images_processed']):
                    raw_resp = img_info.get('result', {}).get('raw_response', 'N/A')
                    
                    # Safe JSON parsing: handle both string and dict formats
                    try:
                        if isinstance(raw_resp, str):
                            # If it's a string, try to parse it
                            if raw_resp.startswith('{'):
                                # Replace Python-style None/True/False with JSON equivalents
                                raw_resp = raw_resp.replace('None', 'null').replace('True', 'true').replace('False', 'false')
                                # Replace single quotes with double quotes (careful with nested quotes)
                                raw_resp = raw_resp.replace("'", '"')
                                parsed = json.loads(raw_resp)
                                st.json(parsed)
                            else:
                                st.text(raw_resp)  # Show as plain text if not JSON
                        elif isinstance(raw_resp, dict):
                            # Already a dict, show directly
                            st.json(raw_resp)
                        else:
                            st.text(str(raw_resp))
                    except json.JSONDecodeError as e:
                        st.warning(f"⚠️ Respuesta #{idx + 1} no es JSON válido")
                        st.code(str(raw_resp), language='python')  # Show as Python code instead
                    except Exception as e:
                        st.error(f"Error mostrando respuesta #{idx + 1}: {e}")
                        st.text(str(raw_resp))
    
    with form_col:
        st.markdown("#### ✏️ Editar Datos Extraídos")
        st.caption("💡 Edita los campos extraídos automáticamente o completa los que falten")
        
        # Info message about auto-save and persistence
        st.info("""
        💾 **Guardado automático activo:**
        - Los cambios se guardan en memoria y en disco automáticamente
        - Permaneces en el mismo activo después de guardar
        - ⚠️ **IMPORTANTE**: Si recargas la página completa (F5), debes:
          1. Ir al tab "📊 Resultados"
          2. Cargar el checkpoint guardado (con tus ediciones)
          3. NO volver a procesar desde Excel (sobrescribiría tus cambios)
        """)
        
        # Get existing results_by_type (data already extracted by API)
        results_by_type = current_result.get('results_by_type', {})
        
        # Get ALL unique types from multiple sources (prioritize extracted data)
        tipos_to_show = {}
        
        # 1. From results_by_type (already processed - PRIORITY)
        for tipo_label in results_by_type.keys():
            tipos_to_show[tipo_label] = {'source': 'extracted', 'data': results_by_type[tipo_label]}
        
        # 2. From images_processed (processed but maybe not in results_by_type?)
        for img_info in current_result.get('images_processed', []):
            tipo_label = img_info.get('type_label', img_info.get('Tipo', 'PLACA1'))
            if tipo_label not in tipos_to_show:
                # Get data from result if available
                result_data = img_info.get('result', {})
                tipos_to_show[tipo_label] = {'source': 'processed', 'data': result_data}
        
        # 3. From selector_df (configured but maybe not processed yet)
        barcode = current_result['BarCode']
        if 'batch_selector_df' in st.session_state and st.session_state.batch_selector_df is not None:
            barcode_rows = st.session_state.batch_selector_df[st.session_state.batch_selector_df['BarCode'] == barcode]
            for idx, row in barcode_rows.iterrows():
                tipo_label = row['Tipo']
                if tipo_label not in tipos_to_show:
                    tipos_to_show[tipo_label] = {'source': 'configured', 'data': {}}
        
        if not tipos_to_show:
            st.warning("⚠️ No hay datos disponibles para este activo")
            st.info("Si acabas de procesar este activo, intenta recargar la página o volver a cargar el checkpoint")
            return
        
        # Show info about available data
        st.caption(f"📊 **{len(tipos_to_show)} tipo(s) de imagen detectados**: {', '.join(tipos_to_show.keys())}")
        
        # Display edit form for EACH type (from extracted, processed, or configured)
        for type_label, tipo_info in tipos_to_show.items():
            # Get existing data from the tipo_info
            data = tipo_info.get('data', {})
            source = tipo_info.get('source', 'unknown')
            
            # Determine if this has extracted data
            has_extracted_data = source == 'extracted' and any([
                data.get('marca'),
                data.get('modelo'),
                data.get('numero_serie'),
                data.get('codigo_scada_principal'),
                data.get('potencia')
            ])
            
            # Icon and label based on data availability
            if has_extracted_data:
                icon = "✅"
                label = f"{icon} {type_label}"
            elif source == 'processed':
                icon = "⚠️"
                label = f"{icon} {type_label} (Datos parciales)"
            else:
                icon = "✏️"
                label = f"{icon} {type_label} (Completar manualmente)"
            
            # Expand by default if no data or if it's the first one
            is_first = list(tipos_to_show.keys()).index(type_label) == 0
            expanded = not has_extracted_data or is_first
            
            with st.expander(label, expanded=expanded):
                with st.form(key=f"review_form_{current_idx}_{type_label}"):
                    
                    # Determine which fields to show based on type
                    is_placa = 'PLACA' in type_label or type_label == 'AMBOS'
                    is_scada = 'SCADA' in type_label or type_label == 'AMBOS'
                    
                    if is_placa:
                        st.markdown("##### 📋 Datos de Placa Técnica")
                        
                        marca = st.text_input(
                            "Marca",
                            value=data.get('marca', '') or '',
                            key=f"marca_{current_idx}_{type_label}",
                            placeholder="Ej: SIEMENS, WEG, ABB..."
                        )
                        
                        modelo = st.text_input(
                            "Modelo",
                            value=data.get('modelo', '') or '',
                            key=f"modelo_{current_idx}_{type_label}",
                            placeholder="Ej: 1LA7133, W22..."
                        )
                        
                        col_pot1, col_pot2 = st.columns(2)
                        with col_pot1:
                            potencia = st.text_input(
                                "Potencia",
                                value=data.get('potencia', '') or '',
                                key=f"potencia_{current_idx}_{type_label}",
                                placeholder="Ej: 5.5KW, 7.5HP, 200CV"
                            )
                        
                        with col_pot2:
                            voltaje = st.text_input(
                                "Voltaje",
                                value=data.get('voltaje', '') or '',
                                key=f"voltaje_{current_idx}_{type_label}",
                                placeholder="Ej: 220/380V"
                            )
                        
                        col_elec1, col_elec2 = st.columns(2)
                        with col_elec1:
                            corriente = st.text_input(
                                "Corriente",
                                value=data.get('corriente', '') or '',
                                key=f"corriente_{current_idx}_{type_label}",
                                placeholder="Ej: 5.2/3.0A"
                            )
                        
                        with col_elec2:
                            frecuencia = st.text_input(
                                "Frecuencia",
                                value=data.get('frecuencia', '') or '',
                                key=f"frecuencia_{current_idx}_{type_label}",
                                placeholder="Ej: 50Hz, 60Hz"
                            )
                        
                        col_mec1, col_mec2 = st.columns(2)
                        with col_mec1:
                            rpm = st.text_input(
                                "RPM",
                                value=data.get('rpm', '') or '',
                                key=f"rpm_{current_idx}_{type_label}",
                                placeholder="Ej: 1440, 1500, 3000"
                            )
                        
                        with col_mec2:
                            frame = st.text_input(
                                "Frame / Carcasa",
                                value=data.get('frame', '') or '',
                                key=f"frame_{current_idx}_{type_label}",
                                placeholder="Ej: 132M, 100L"
                            )
                        
                        numero_serie = st.text_input(
                            "Número de Serie",
                            value=data.get('numero_serie', '') or '',
                            key=f"sn_{current_idx}_{type_label}",
                            placeholder="Ej: 12345678"
                        )
                        
                        año_value = data.get('año', None)
                        try:
                            año_value = int(año_value) if año_value else None
                        except:
                            año_value = None
                        
                        col_fecha1, col_fecha2 = st.columns(2)
                        with col_fecha1:
                            año = st.number_input(
                                "Año de Fabricación",
                                min_value=1950,
                                max_value=2025,
                                value=año_value,
                                step=1,
                                key=f"año_{current_idx}_{type_label}",
                                help="Dejar vacío si no aplica"
                            )
                        
                        with col_fecha2:
                            fecha_fabricacion = st.text_input(
                                "Fecha Completa",
                                value=data.get('fecha_fabricacion', '') or '',
                                key=f"fecha_{current_idx}_{type_label}",
                                placeholder="Ej: 03/2015"
                            )
                        
                        # Additional fields
                        col_add1, col_add2 = st.columns(2)
                        with col_add1:
                            ip_protection = st.text_input(
                                "IP / Protección",
                                value=data.get('ip_protection', '') or '',
                                key=f"ip_{current_idx}_{type_label}",
                                placeholder="Ej: IP55, IP65"
                            )
                        
                        with col_add2:
                            insulation_class = st.text_input(
                                "Clase Aislamiento",
                                value=data.get('insulation_class', '') or '',
                                key=f"insulation_{current_idx}_{type_label}",
                                placeholder="Ej: F, B, H"
                            )
                        
                        observaciones = st.text_area(
                            "Observaciones",
                            value=data.get('observaciones', '') or '',
                            key=f"obs_{current_idx}_{type_label}",
                            placeholder="Cualquier dato adicional relevante...",
                            height=80
                        )
                    
                    # Show SCADA fields if applicable
                    if is_scada:
                        if is_placa:  # If AMBOS, add separator
                            st.markdown("---")
                        
                        st.markdown("##### 🔢 Código SCADA")
                        codigo_scada = st.text_input(
                            "Código SCADA Principal",
                            value=data.get('codigo_scada_principal', '') or '',
                            key=f"scada_{current_idx}_{type_label}",
                            placeholder="Ej: MOT-001, PUMP-123"
                        )
                        
                        codigo_scada_alt = st.text_input(
                            "Código SCADA Alternativo",
                            value=data.get('codigo_scada_alternativo', '') or '',
                            key=f"scada_alt_{current_idx}_{type_label}",
                            placeholder="Código secundario si existe"
                        )
                        
                        if not is_placa:  # Only if NOT AMBOS (avoid duplicate observaciones)
                            observaciones_scada = st.text_area(
                                "Observaciones",
                                value=data.get('observaciones', '') or '',
                                key=f"obs_scada_{current_idx}_{type_label}",
                                placeholder="Cualquier dato adicional del código SCADA...",
                                height=80
                            )
                    
                    # Submit button
                    submitted = st.form_submit_button("💾 Guardar", type="primary", use_container_width=True)
                    
                    if submitted:
                        # Build updated_data based on what fields were shown
                        updated_data = {}
                        
                        if is_placa:
                            updated_data.update({
                                'marca': marca if marca else None,
                                'modelo': modelo if modelo else None,
                                'potencia': potencia if potencia else None,
                                'voltaje': voltaje if voltaje else None,
                                'corriente': corriente if corriente else None,
                                'frecuencia': frecuencia if frecuencia else None,
                                'rpm': rpm if rpm else None,
                                'frame': frame if frame else None,
                                'numero_serie': numero_serie if numero_serie else None,
                                'año': año if año else None,
                                'fecha_fabricacion': fecha_fabricacion if fecha_fabricacion else None,
                                'ip_protection': ip_protection if ip_protection else None,
                                'insulation_class': insulation_class if insulation_class else None,
                                'observaciones': observaciones if observaciones else None,
                            })
                        
                        if is_scada:
                            updated_data.update({
                                'codigo_scada_principal': codigo_scada if codigo_scada else None,
                                'codigo_scada_alternativo': codigo_scada_alt if codigo_scada_alt else None,
                            })
                            if not is_placa:  # Only if SCADA-only (avoid overwriting observaciones from PLACA)
                                updated_data['observaciones'] = observaciones_scada if observaciones_scada else None
                        
                        # Add metadata
                        updated_data['confidence'] = data.get('confidence', 0.0)
                        updated_data['raw_response'] = data.get('raw_response')
                        updated_data['manual_edit'] = True
                        
                        # Save to batch_results (using current_activo_cache_idx for correct position)
                        actual_idx = st.session_state.current_activo_cache_idx
                        
                        # Ensure results_by_type exists
                        if 'results_by_type' not in st.session_state.batch_results[actual_idx]:
                            st.session_state.batch_results[actual_idx]['results_by_type'] = {}
                        
                        st.session_state.batch_results[actual_idx]['results_by_type'][type_label] = updated_data
                        
                        # Update cache immediately to reflect changes
                        if 'results_by_type' not in st.session_state.current_activo_cache:
                            st.session_state.current_activo_cache['results_by_type'] = {}
                        st.session_state.current_activo_cache['results_by_type'][type_label] = updated_data
                        
                        # CRITICAL: Force persist current index - set it MULTIPLE times to ensure it sticks
                        st.session_state.review_index = actual_idx
                        st.session_state['_review_index_backup'] = actual_idx  # Backup
                        
                        # Track last edit for results table notification
                        st.session_state['last_edited_activo_idx'] = actual_idx
                        st.session_state['last_edited_activo_type'] = type_label
                        
                        # Auto-save to disk
                        if st.session_state.get('last_batch_results_file'):
                            try:
                                with open(st.session_state.last_batch_results_file, 'w', encoding='utf-8') as f:
                                    json.dump(st.session_state.batch_results, f, indent=2, ensure_ascii=False, default=str)
                                logger.info(f"✅ Auto-saved after edit: activo #{actual_idx + 1} → {st.session_state.last_batch_results_file}")
                            except Exception as e:
                                logger.error(f"❌ Failed to auto-save after edit: {e}")
                        
                        # Mark results table as needing update
                        st.session_state['results_table_updated'] = True
                        
                        logger.info(f"✅ Guardado: Activo #{actual_idx + 1} [{type_label}] - review_index={st.session_state.review_index}")
                        st.success(f"✅ {type_label} guardado en activo #{actual_idx + 1} - Tabla de resultados actualizada")
                        
                        # NO hacer st.rerun() - dejar que Streamlit lo maneje naturalmente
                        # El formulario se limpia automáticamente después del submit
    
    # Progress indicator
    st.markdown("---")
    progress = (current_idx + 1) / total_results
    st.progress(progress)
    
    reviewed_count = sum(1 for r in st.session_state.batch_results if r.get('reviewed', False))
    st.caption(f"Revisados: {reviewed_count}/{total_results} activos")


def main():
    initialize_session_state()
    
    # === SIDEBAR MONITOR - RENDER FIRST (always visible) ===
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Queue Monitor at top
        st.markdown("---")
        st.markdown("### 📊 Cola de Procesamiento")
        
        current_queue = st.session_state.get('processing_queue', [])
        queue_length = len(current_queue)
        
        if queue_length > 0:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 12px; 
                        border-radius: 8px; 
                        margin-bottom: 10px;'>
                <p style='color: white; margin: 0; font-weight: bold; font-size: 16px;'>
                    ⏳ {queue_length} imágenes en cola
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show compact queue
            for i, row_key in enumerate(current_queue[:5], 1):
                result_info = st.session_state.individual_results.get(row_key, {})
                barcode = result_info.get('barcode', 'N/A')
                tipo = result_info.get('tipo_label', 'N/A')
                
                if i == 1:
                    st.markdown(f"🔄 **#{i}**: `{barcode}` [{tipo}]")
                else:
                    st.caption(f"⏳ **#{i}**: `{barcode}` [{tipo}]")
            
            if queue_length > 5:
                st.caption(f"... y {queue_length - 5} más")
            
            st.caption(f"🕐 Actualizado: {datetime.now().strftime('%H:%M:%S')}")
            
            if st.button("🗑️ Limpiar Cola", use_container_width=True, key="sidebar_clear_queue_top"):
                st.session_state.processing_queue.clear()
                st.warning("⚠️ Cola limpiada")
                st.rerun()
        else:
            st.success("✅ Cola vacía")
        
        st.markdown("---")
    
    # === PROCESS QUEUE ITEMS ===
    queue_length_before = len(st.session_state.get('processing_queue', []))
    
    if queue_length_before > 0:
        logger.info(f"=== MAIN START: Queue has {queue_length_before} items, processing first item...")
        process_queued_items()
    
    # Check if more items remain after processing
    queue_length_after = len(st.session_state.get('processing_queue', []))
    
    # If we just processed an item and there are more in queue, rerun immediately
    if queue_length_before > 0 and queue_length_after > 0:
        logger.info(f"=== MAIN END: Processed 1 item, {queue_length_after} remaining. Scheduling auto-rerun...")
        time.sleep(0.3)  # Small delay to prevent browser overload
        st.rerun()
    elif queue_length_before > 0 and queue_length_after == 0:
        logger.info(f"=== MAIN END: Queue now empty. All items processed.")
    else:
        logger.debug("=== MAIN: No queue processing needed")
    
    st.title("⚡ Procesamiento Rápido en Batch")
    st.caption("Procesa múltiples imágenes de forma rápida y luego revisa/corrige en la transcripción asistida")
    
    # Show checkpoint restoration notification
    if st.session_state.get('checkpoint_restored'):
        checkpoint_info = st.session_state.checkpoint_restored
        queue_len = checkpoint_info.get('queue_len', 0)
        results_len = checkpoint_info.get('results_len', 0)
        timestamp = checkpoint_info.get('timestamp', 'unknown')
        
        if queue_len > 0 or results_len > 0:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 15px; 
                        border-radius: 10px; 
                        margin-bottom: 15px;
                        border-left: 5px solid #e91e63;'>
                <h3 style='color: white; margin: 0;'>🔄 CHECKPOINT RESTAURADO</h3>
                <p style='color: white; margin: 5px 0 0 0; font-size: 14px;'>
                    Sesión recuperada del {timestamp[:19].replace('T', ' ')}:<br>
                    • <strong>{results_len} imágenes procesadas</strong><br>
                    • <strong>{queue_len} imágenes en cola pendientes</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("✅ Puedes continuar desde donde lo dejaste o limpiar para empezar de nuevo.")
            with col2:
                if st.button("🗑️ Limpiar y Empezar", use_container_width=True, type="secondary"):
                    clear_checkpoint_incremental()
                    st.rerun()
    
    # Show simple banner if there are items in queue
    if st.session_state.get('processing_queue') and len(st.session_state.processing_queue) > 0:
        st.info(f"⏳ **{len(st.session_state.processing_queue)} imágenes en cola de procesamiento** - Ver detalles en el panel lateral ⬅️")
    
    # Continue sidebar (API key check)
    with st.sidebar:
        # API Key check
        api_key = os.getenv('OPENAI_API_KEY', '')
        if api_key:
            st.success("✅ API Key configurada")
        else:
            st.error("❌ API Key no configurada")
            st.info("Configura tu API key en la página de Transcripción Asistida")
        
        st.markdown("---")
        
        # File uploader
        excel_file = st.file_uploader(
            "Cargar archivo Excel",
            type=['xlsx', 'xls'],
            help="Excel con columnas 'BarCode' e imágenes",
            key="batch_excel_upload"
        )
        
        if excel_file:
            # Save uploaded file
            excel_path = Path(st.session_state.batch_config.output_dir) / "uploaded_excel_batch.xlsx"
            with open(excel_path, 'wb') as f:
                f.write(excel_file.getvalue())
            
            # Only load data if not already loaded (avoid re-extraction if checkpoint already loaded)
            if st.session_state.batch_data_df is None or st.session_state.get('excel_path') != str(excel_path):
                output_dir = Path(st.session_state.batch_config.output_dir)
                st.session_state.batch_data_df = load_excel_data_batch(str(excel_path), output_dir)
                st.session_state.excel_path = str(excel_path)
            else:
                logger.info("⏭️ Excel data already loaded - skipping re-extraction")
            
            # Try to load previous configuration for this Excel file
            excel_name = Path(excel_file.name).stem
            config_file = find_configuration_file(excel_name)
            
            if config_file:
                st.info(f"📂 Se encontró una configuración previa para este archivo: {config_file.name}")
                if st.button("📥 Cargar Configuración Previa", key="load_prev_config"):
                    loaded_config = load_configuration_from_file(config_file)
                    if loaded_config is not None:
                        st.session_state.batch_selector_df = loaded_config
                        st.session_state.batch_df_needs_refresh = False
                        st.success("✅ Configuración cargada correctamente!")
                        st.rerun()
    
    # Main content - Tab3 (Resultados) always available even without Excel
    # This allows loading checkpoints directly without re-extracting images
    tab1, tab2, tab3 = st.tabs(["📋 Selección", "🔍 Revisión", "📊 Resultados"])
    
    # Tabs 1 & 2 require Excel data
    if st.session_state.batch_data_df is not None:
        df = st.session_state.batch_data_df
        
        with tab1:
            total_pending, default_method, default_model, processed_count = display_batch_row_selector(df)
            
            st.markdown("---")
            
            # ==== SISTEMA DE LOTES AUTOMÁTICO ====
            if total_pending > 500:
                st.markdown("### 🔥 Sistema de Procesamiento por Lotes")
                st.warning(f"⚠️ **Dataset grande detectado: {total_pending} imágenes pendientes** ({processed_count} ya procesadas) - Procesamiento automático en lotes recomendado")
                
                # Batch size configuration
                batch_col1, batch_col2 = st.columns([3, 1])
                with batch_col1:
                    batch_size = st.slider(
                        "Tamaño de lote (imágenes por lote)",
                        min_value=100,
                        max_value=500,
                        value=400,
                        step=50,
                        help="Cantidad de imágenes a procesar en cada lote. Recomendado: 300-500"
                    )
                with batch_col2:
                    st.metric("Lotes necesarios", f"{(total_pending + batch_size - 1) // batch_size}")
                
                # Generate batches from configuration
                selected_df = st.session_state.batch_selector_df[st.session_state.batch_selector_df['✅'] == True].copy()
                
                # Create batches
                batches = []
                for i in range(0, len(selected_df), batch_size):
                    batch_df = selected_df.iloc[i:i+batch_size]
                    
                    # Count by type
                    tipo_counts = batch_df['Tipo'].value_counts().to_dict()
                    
                    batches.append({
                        'batch_num': len(batches) + 1,
                        'start_idx': i,
                        'end_idx': min(i + batch_size, len(selected_df)),
                        'size': len(batch_df),
                        'df': batch_df,
                        'tipo_counts': tipo_counts
                    })
                
                # Store in session state
                if 'batch_list' not in st.session_state:
                    st.session_state.batch_list = batches
                else:
                    st.session_state.batch_list = batches  # Update if configuration changed
                
                # Display batches with individual process buttons
                st.markdown("#### 📦 Lotes configurados:")
                
                for batch in batches:
                    with st.expander(
                        f"**Lote {batch['batch_num']}** - {batch['size']} imágenes (filas {batch['start_idx']+1} a {batch['end_idx']})",
                        expanded=(batch['batch_num'] == 1)  # First batch expanded by default
                    ):
                        # Show composition
                        tipo_summary = " | ".join([f"{tipo}: {count}" for tipo, count in batch['tipo_counts'].items()])
                        st.caption(f"📊 Composición: {tipo_summary}")
                        
                        # Time estimate
                        estimated_time = batch['size'] * 2.5 / 60  # minutes
                        st.caption(f"⏱️ Tiempo estimado: {estimated_time:.0f} minutos")
                        
                        # Process button for this batch
                        batch_button_col1, batch_button_col2 = st.columns([3, 1])
                        
                        with batch_button_col1:
                            # Create unique key for button
                            button_key = f"process_batch_{batch['batch_num']}"
                            
                            if st.button(
                                f"🚀 Procesar Lote {batch['batch_num']}",
                                key=button_key,
                                use_container_width=True,
                                type="primary"
                            ):
                                # Check API key if needed
                                if default_method == "API OpenAI" and not os.getenv('OPENAI_API_KEY'):
                                    st.error("⚠️ API Key de OpenAI no configurada.")
                                else:
                                    # Set processing flag to prevent page reload
                                    st.session_state[f'processing_batch_{batch["batch_num"]}'] = True
                                    
                                    # Show immediate feedback
                                    with st.spinner(f"Iniciando procesamiento del Lote {batch['batch_num']}..."):
                                        time.sleep(0.5)  # Brief pause to show spinner
                                    
                                    # Process this batch
                                    try:
                                        process_specific_batch(batch, default_method, default_model)
                                        st.session_state[f'processing_batch_{batch["batch_num"]}'] = False
                                    except Exception as e:
                                        st.error(f"❌ Error procesando lote: {e}")
                                        logger.error(f"Batch processing error: {e}")
                                        st.session_state[f'processing_batch_{batch["batch_num"]}'] = False
                        
                        with batch_button_col2:
                            if st.session_state.batch_results:
                                if st.button("💾 Exportar", key=f"export_batch_{batch['batch_num']}", use_container_width=True):
                                    export_results()
                
                st.markdown("---")
                
                # ==== AUTO-PROCESS ALL BATCHES BUTTON ====
                st.markdown("### 🎯 Procesamiento Automático")
                
                # DETECT COMPLETED BATCHES from individual_results
                completed_batches = []
                pending_batches = []
                
                if st.session_state.get('individual_results'):
                    # Check which batches are fully processed
                    for batch in batches:
                        batch_df = batch['df']
                        batch_images_processed = 0
                        
                        for _, row in batch_df.iterrows():
                            barcode = row['BarCode']
                            tipo_label = row['Tipo']
                            img_idx = row.get('_img_idx', '')
                            row_key = f"{barcode}_{tipo_label}_{img_idx}"
                            
                            if row_key in st.session_state.individual_results:
                                result = st.session_state.individual_results[row_key]
                                if result.get('status') == 'success':
                                    batch_images_processed += 1
                        
                        completion_rate = (batch_images_processed / batch['size']) * 100 if batch['size'] > 0 else 0
                        
                        if completion_rate >= 99:  # Consider complete if 99%+ done
                            completed_batches.append(batch['batch_num'])
                        else:
                            pending_batches.append({
                                'num': batch['batch_num'],
                                'processed': batch_images_processed,
                                'total': batch['size'],
                                'rate': completion_rate
                            })
                else:
                    # No individual_results, all batches are pending
                    pending_batches = [{'num': b['batch_num'], 'processed': 0, 'total': b['size'], 'rate': 0.0} for b in batches]
                
                total_batches = len(batches)
                total_completed = len(completed_batches)
                total_pending_batches = len(pending_batches)
                
                # Calculate realistic time estimate (only for pending batches)
                total_time_estimate = sum(
                    (p['total'] - p['processed']) * 2.5 / 60 
                    for p in pending_batches
                )
                
                # Show status
                if total_completed > 0:
                    st.success(f"✅ **{total_completed}/{total_batches} lotes ya completados** - El procesamiento automático continuará desde donde quedó")
                    
                    if pending_batches:
                        st.info(f"🔄 **{total_pending_batches} lotes pendientes de procesar:**")
                        for p in pending_batches[:3]:  # Show first 3 pending
                            st.caption(f"  • Lote {p['num']}: {p['processed']}/{p['total']} imágenes ({p['rate']:.0f}% completo)")
                        if len(pending_batches) > 3:
                            st.caption(f"  ... y {len(pending_batches) - 3} lotes más")
                else:
                    st.info("🚀 **Procesa todos los lotes secuencialmente** sin intervención manual. El sistema procesará cada lote, guardará checkpoints, y continuará con el siguiente automáticamente.")
                
                auto_col1, auto_col2, auto_col3 = st.columns([2, 1, 1])
                
                with auto_col1:
                    # Smart button text
                    if total_completed > 0:
                        button_text = f"🔄 CONTINUAR PROCESAMIENTO ({total_pending_batches} LOTES PENDIENTES)"
                        button_help = f"Procesa solo los {total_pending_batches} lotes pendientes. Saltará automáticamente los {total_completed} lotes ya completados."
                    else:
                        button_text = f"🚀 PROCESAR TODOS LOS {total_batches} LOTES AUTOMÁTICAMENTE"
                        button_help = f"Procesa los {total_batches} lotes secuencialmente. Tiempo estimado: {total_time_estimate:.0f} min"
                    
                    if st.button(
                        button_text,
                        use_container_width=True,
                        type="primary",
                        help=button_help
                    ):
                        if default_method == "API OpenAI" and not os.getenv('OPENAI_API_KEY'):
                            st.error("⚠️ API Key de OpenAI no configurada.")
                        else:
                            # Process all batches sequentially (skip completed ones)
                            st.session_state.auto_processing_all = True
                            
                            # Create progress containers
                            overall_progress = st.progress(0.0)
                            overall_status = st.empty()
                            batch_status = st.empty()
                            
                            start_time_all = time.time()
                            batches_processed = 0
                            batches_skipped = 0
                            
                            for idx, batch in enumerate(batches):
                                batch_num = batch['batch_num']
                                
                                # SKIP if batch already completed
                                if batch_num in completed_batches:
                                    batches_skipped += 1
                                    overall_progress.progress((idx + 1) / total_batches)
                                    overall_status.info(f"⏭️ Lote {batch_num} ya completado - Saltando... ({batches_processed} procesados, {batches_skipped} saltados)")
                                    logger.info(f"⏭️ Skipping Lote {batch_num} - already completed (99%+)")
                                    time.sleep(0.5)  # Brief pause to show message
                                    continue
                                
                                # Update overall progress
                                overall_progress.progress((idx) / total_batches)
                                overall_status.info(f"📦 Procesando Lote {batch_num}/{total_batches}... ({batches_processed} procesados, {batches_skipped} saltados)")
                                
                                try:
                                    # Show batch info
                                    pending_info = next((p for p in pending_batches if p['num'] == batch_num), None)
                                    if pending_info:
                                        images_to_process = pending_info['total'] - pending_info['processed']
                                        batch_status.markdown(f"""
                                        **Lote {batch_num}**: {images_to_process}/{batch['size']} imágenes pendientes ({pending_info['rate']:.0f}% completo)  
                                        📊 Composición: {' | '.join([f"{t}: {c}" for t, c in batch['tipo_counts'].items()])}  
                                        ⏱️ Tiempo estimado: {images_to_process * 2.5 / 60:.0f} min
                                        """)
                                    else:
                                        batch_status.markdown(f"""
                                        **Lote {batch_num}**: {batch['size']} imágenes  
                                        📊 Composición: {' | '.join([f"{t}: {c}" for t, c in batch['tipo_counts'].items()])}  
                                        ⏱️ Tiempo estimado: {batch['size'] * 2.5 / 60:.0f} min
                                        """)
                                    
                                    # Process this batch (anti-reprocesamiento protection active)
                                    process_specific_batch(batch, default_method, default_model)
                                    batches_processed += 1
                                    
                                    # Update progress
                                    overall_progress.progress((idx + 1) / total_batches)
                                    
                                    # Brief pause between batches
                                    if idx < total_batches - 1:
                                        time.sleep(1)
                                    
                                except Exception as e:
                                    st.error(f"❌ Error en Lote {batch_num}: {e}")
                                    logger.error(f"Auto-processing failed at batch {batch_num}: {e}")
                                    
                                    # Ask user if they want to continue
                                    if st.button(f"⏭️ Continuar con Lote {batch_num + 1}", key=f"continue_after_error_{batch_num}"):
                                        continue
                                    else:
                                        break
                            
                            # Final summary
                            elapsed_all = time.time() - start_time_all
                            overall_progress.progress(1.0)
                            overall_status.success(f"✅ ¡Procesamiento automático completado!")
                            batch_status.markdown(f"""
                            ### 🎉 Resumen Final
                            - **Lotes procesados**: {batches_processed} nuevos + {batches_skipped} saltados = {total_batches} totales
                            - **Total de activos**: {len(st.session_state.batch_results)}
                            - **Tiempo total**: {elapsed_all / 60:.1f} minutos
                            - **Checkpoints guardados**: ✅ Cada lote guardado automáticamente
                            
                            💡 Los resultados están listos para revisar en la pestaña "📋 Revisar Resultados"
                            """)
                            
                            st.session_state.auto_processing_all = False
                            st.balloons()
                
                with auto_col2:
                    if total_completed > 0:
                        st.metric("Pendientes", f"{total_pending_batches}/{total_batches}")
                    else:
                        st.metric("Total Lotes", total_batches)
                
                with auto_col3:
                    st.metric("Tiempo Est.", f"{total_time_estimate:.0f} min")
                
                if total_completed > 0:
                    st.caption(f"✅ {total_completed} lotes completados serán saltados automáticamente")
                    st.caption(f"🔄 {total_pending_batches} lotes pendientes × ~{total_time_estimate/max(total_pending_batches,1):.0f} min/lote = {total_time_estimate:.0f} min totales")
                else:
                    st.caption(f"⚡ El sistema procesará {total_batches} lotes × {batch_size} imágenes/lote = {total_pending} imágenes totales")
                st.caption("💾 Se guardarán checkpoints cada 50 imágenes + checkpoint final por lote")
                st.caption("🛡️ Anti-reprocesamiento activo: Saltará imágenes ya procesadas automáticamente")
                
                st.markdown("---")
                st.info("💡 **Tip Alternativo**: También puedes procesar lote por lote manualmente usando los botones individuales arriba.")
                
                # Test connectivity button
                st.markdown("---")
                st.markdown("### 🔧 Herramientas de Diagnóstico")
                
                test_col1, test_col2 = st.columns([2, 2])
                with test_col1:
                    if st.button("🔍 Test de Conectividad API", use_container_width=True, help="Verifica que la API de OpenAI funciona correctamente"):
                        with st.spinner("Verificando conexión..."):
                            try:
                                # Initialize API
                                config_obj = st.session_state.batch_config
                                api_config = config_obj.config.get('api', {}).copy() if hasattr(config_obj, 'config') else {}
                                if default_model:
                                    api_config['model'] = default_model
                                
                                test_extractor = APIExtractor(api_config)
                                
                                # Test 1: List models
                                st.write("**Test 1:** Listando modelos disponibles...")
                                models = test_extractor.client.models.list()
                                st.success(f"✅ Conexión exitosa - {len(models.data)} modelos disponibles")
                                
                                # Test 2: Get a sample image from configuration
                                st.write("**Test 2:** Probando extracción con una imagen de prueba...")
                                test_df = st.session_state.batch_selector_df[st.session_state.batch_selector_df['✅'] == True]
                                if len(test_df) > 0:
                                    test_row = test_df.iloc[0]
                                    test_path = test_row['_img_path']
                                    test_barcode = test_row['BarCode']
                                    test_tipo = test_row['Tipo']
                                    
                                    # Map tipo
                                    tipo_mapping = {
                                        'PLACA1': 'placa_tecnica', 'PLACA2': 'placa_tecnica', 'PLACA3': 'placa_tecnica',
                                        'SCADA1': 'codigo_scada', 'SCADA2': 'codigo_scada', 'SCADA3': 'codigo_scada',
                                        'AMBOS': None
                                    }
                                    test_image_type = tipo_mapping.get(test_tipo)
                                    
                                    start_test = time.time()
                                    test_result = test_extractor.extract_from_image_assisted(
                                        test_path, test_barcode, None, test_image_type
                                    )
                                    test_duration = time.time() - start_test
                                    
                                    st.success(f"✅ Extracción exitosa en {test_duration:.1f}s")
                                    st.write(f"- **BarCode:** {test_barcode}")
                                    st.write(f"- **Tipo:** {test_tipo}")
                                    if 'PLACA' in test_tipo:
                                        st.write(f"- **Marca:** {test_result.get('marca', 'N/A')}")
                                        st.write(f"- **Modelo:** {test_result.get('modelo', 'N/A')}")
                                    elif 'SCADA' in test_tipo:
                                        st.write(f"- **Código SCADA:** {test_result.get('codigo_scada_principal', 'N/A')}")
                                    st.write(f"- **Confianza:** {test_result.get('overall_confidence', 0):.0%}")
                                    
                                    st.info(f"💡 Tiempo estimado para {total_pending} imágenes: ~{(total_pending * test_duration / 60):.0f} minutos")
                                else:
                                    st.warning("⚠️ No hay imágenes seleccionadas para probar")
                                
                            except Exception as e:
                                st.error(f"❌ Error en test: {e}")
                                st.error("Verifica tu API Key y conexión a internet")
                                logger.error(f"Connectivity test failed: {e}")
                
                with test_col2:
                    if st.button("📊 Ver Configuración Actual", use_container_width=True, help="Muestra la configuración actual del lote"):
                        st.write("**Resumen de configuración:**")
                        st.write(f"- **Total imágenes seleccionadas:** {total_pending + processed_count} ({processed_count} ya procesadas)")
                        st.write(f"- **Imágenes pendientes:** {total_pending}")
                        st.write(f"- **Método:** {default_method}")
                        st.write(f"- **Modelo:** {default_model}")
                        
                        # Count by type
                        selected_df = st.session_state.batch_selector_df[st.session_state.batch_selector_df['✅'] == True]
                        tipo_counts = selected_df['Tipo'].value_counts().to_dict()
                        st.write("**Distribución por tipo:**")
                        for tipo, count in tipo_counts.items():
                            st.write(f"  - {tipo}: {count} imágenes")
            
            # Process button - BIG and prominent (for smaller datasets or manual processing)
            if total_pending > 0 and total_pending <= 500:
                st.markdown("### 🚀 ¿Listo para procesar?")
                
                # Show clear breakdown
                if processed_count > 0:
                    st.info(f"📋 Se procesarán **{total_pending} imágenes pendientes** (el sistema saltará {processed_count} ya procesadas automáticamente)")
                else:
                    st.info(f"📋 Se procesarán **{total_pending} imágenes**")
                
                # Validation warnings
                if default_method == "API OpenAI" and not os.getenv('OPENAI_API_KEY'):
                    st.error("⚠️ **API Key de OpenAI no configurada.** Ve a la página de Transcripción Asistida para configurarla.")
                    st.info("💡 **Tip:** También puedes configurarla en un archivo `.env` con la variable `OPENAI_API_KEY=tu_clave_aqui`")
                    can_process = False
                else:
                    can_process = True
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(
                        f"🚀 PROCESAR {total_pending} IMÁGENES PENDIENTES (TODAS LAS PÁGINAS)",
                        use_container_width=True,
                        type="primary",
                        disabled=not can_process,
                        help=f"Procesa {total_pending} imágenes pendientes de TODAS las páginas (salta {processed_count} ya procesadas)"
                    ):
                        process_batch(default_method, default_model)
                
                with col2:
                    if st.session_state.batch_results:
                        if st.button("💾 Exportar", use_container_width=True):
                            export_results()
            else:
                st.info("👆 Selecciona al menos una imagen para procesar. Usa los botones de 'Selección Rápida' para marcar todas de una vez.")
        
        with tab2:
            # Review interface
            display_review_interface()
        
        # tab3 is handled OUTSIDE this if block (see below) to allow checkpoint loading without Excel
    else:
        # No Excel loaded - show placeholder in tabs 1 & 2
        with tab1:
            st.info("👆 Sube un archivo Excel en la barra lateral para comenzar la configuración de procesamiento")
        
        with tab2:
            st.info("👆 Sube un archivo Excel y procesa imágenes para usar la interfaz de revisión")
    
    # === TAB3: RESULTADOS - Always accessible regardless of Excel loaded ===
    # This allows loading checkpoints directly without re-extracting images from Excel
    with tab3:
        logger.info("🔍 TAB3 ENTERED")
        logger.info(f"🔍 batch_results exists = {'batch_results' in st.session_state}")
        logger.info(f"🔍 batch_results length = {len(st.session_state.get('batch_results', []))}")
        
        # Check if there are individual results that need consolidation
        has_individual_results = 'individual_results' in st.session_state and st.session_state.individual_results
        successful_individual = 0
        if has_individual_results:
            successful_individual = sum(1 for v in st.session_state.individual_results.values() if v.get('status') == 'success')
        
        # Show sync status - detect mismatch between individual_results and batch_results
        if has_individual_results and successful_individual > 0:
            current_batch_count = len(st.session_state.batch_results) if st.session_state.batch_results else 0
            
            # Count unique barcodes in individual_results
            unique_barcodes = set()
            for row_key in st.session_state.individual_results:
                if st.session_state.individual_results[row_key].get('status') == 'success':
                    # Extract barcode from row_key (format: barcode_tipo_imgidx)
                    parts = row_key.split('_')
                    if len(parts) >= 2:
                        unique_barcodes.add(parts[0])
            
            expected_batch_count = len(unique_barcodes)
            
            # Detect mismatch
            if expected_batch_count != current_batch_count:
                st.warning(f"⚠️ **Desincronización detectada:** {successful_individual} imágenes procesadas → {expected_batch_count} activos esperados, pero solo hay {current_batch_count} en resultados")
                st.info("💡 Esto puede ocurrir si cargaste un checkpoint mientras había resultados antiguos. Usa 'Forzar Reconstrucción' para corregirlo.")
            else:
                st.info(f"ℹ️ **Procesamiento individual detectado:** {successful_individual} imágenes procesadas desde la tabla")
            
            # Consolidation buttons
            sync_col1, sync_col2, sync_col3 = st.columns([2, 2, 1])
            with sync_col1:
                if st.button("🔄 Sincronizar", type="secondary", use_container_width=True, key="sync_individual_to_batch_unified", help="Actualiza el conteo sin modificar datos"):
                    # Just trigger recount
                    logger.info("🔄 Manual sync triggered from tab3 (recount only)")
                    st.success(f"✅ Sincronización completada - {len(st.session_state.batch_results)} activos en resultados")
                    st.rerun()
            
            with sync_col2:
                if st.button("🔨 Forzar Reconstrucción", type="primary", use_container_width=True, key="force_rebuild_batch", help="Reconstruye batch_results desde individual_results (elimina resultados antiguos)"):
                    # FORCE rebuild from individual_results - this REPLACES old batch_results
                    logger.info("🔨 Force rebuilding batch_results from individual_results...")
                    logger.info(f"   BEFORE: {len(st.session_state.batch_results)} batch_results")
                    logger.info(f"   BEFORE: {len(st.session_state.individual_results)} individual_results")
                    
                    # Clear old batch_results
                    st.session_state.batch_results = []
                    logger.info(f"   CLEARED: batch_results is now empty")
                    
                    # Rebuild from individual_results
                    reconstructed = reconstruct_batch_results_from_individual()
                    logger.info(f"   AFTER: {len(st.session_state.batch_results)} batch_results (reconstructed {reconstructed})")
                    
                    if reconstructed > 0:
                        # Save the reconstructed state as checkpoint
                        try:
                            save_checkpoint_incremental()
                            logger.info("✅ Reconstructed state saved as checkpoint")
                        except Exception as save_err:
                            logger.warning(f"Could not save checkpoint after reconstruction: {save_err}")
                        
                        st.success(f"✅ Reconstruidos {reconstructed} activos desde {successful_individual} imágenes procesadas")
                        logger.info(f"✅ Force rebuild complete: {reconstructed} activos")
                    else:
                        st.warning("⚠️ No se pudieron reconstruir resultados - verifica que individual_results tenga datos válidos")
                    
                    time.sleep(1)
                    st.rerun()
            
            with sync_col3:
                st.markdown(f"<div style='text-align: center; padding: 10px; background-color: #e3f2fd; border-radius: 5px;'><strong>{current_batch_count}</strong><br/>activos</div>", unsafe_allow_html=True)
            
            st.markdown("---")
        
        # Option to load previous results (works with or without Excel loaded!)
        if not st.session_state.batch_results:
            st.markdown("### 📂 Cargar Resultados Previos")
            
            # Show helpful message if no Excel loaded
            if st.session_state.batch_data_df is None:
                st.info("💡 **Tip:** Puedes cargar checkpoints directamente sin necesidad de cargar el Excel primero - ¡los resultados se mostrarán instantáneamente!")
            
            previous_files = load_previous_results()
            
            if previous_files:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    file_options = {f.name: f for f in previous_files}
                    selected_file_name = st.selectbox(
                        "Selecciona un archivo de resultados:",
                        options=list(file_options.keys()),
                        key="load_results_tab3_unified"
                    )
                
                with col2:
                    if st.button("📥 Cargar", use_container_width=True, key="load_results_tab3_btn_unified"):
                        selected_file = file_options[selected_file_name]
                        load_results_from_file(selected_file)
                        st.rerun()
            else:
                if st.session_state.batch_data_df is None:
                    st.info("No hay resultados previos. Primero carga un Excel y procesa imágenes en la pestaña 'Selección'")
                else:
                    st.info("No hay resultados previos. Procesa imágenes en la pestaña 'Selección' o desde la tabla de Revisión.")
        
        if st.session_state.batch_results:
            st.markdown("### 📊 Resultados del Procesamiento Batch")
            
            # Summary metrics
            total_activos = len(st.session_state.batch_results)
            total_procesadas = sum(len(r.get('images_processed', [])) for r in st.session_state.batch_results)
            total_exitosos = sum(1 for r in st.session_state.batch_results if r.get('status') == 'success')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Activos Procesados", total_activos)
            with col2:
                st.metric("Imágenes Procesadas", total_procesadas)
            with col3:
                st.metric("Exitosos", f"{total_exitosos}/{total_activos}")
            
            st.markdown("---")
            
            # Show update notification if results were just modified
            if st.session_state.get('results_table_updated', False):
                last_edited_idx = st.session_state.get('last_edited_activo_idx')
                last_edited_type = st.session_state.get('last_edited_activo_type', '')
                if last_edited_idx is not None:
                    barcode = st.session_state.batch_results[last_edited_idx].get('BarCode', 'N/A')
                    st.info(f"🔄 **Actualización en tiempo real:** Activo #{last_edited_idx + 1} (BarCode: {barcode}, Tipo: {last_edited_type}) - Los cambios ya están visibles en la tabla ⬇️")
                else:
                    st.success("✅ Tabla actualizada con los últimos cambios")
                st.session_state['results_table_updated'] = False  # Reset flag
            
            # Quick search for edited activo
            search_col1, search_col2 = st.columns([3, 1])
            with search_col1:
                search_barcode = st.text_input("🔍 Buscar por BarCode", key="search_results_barcode", placeholder="Ingresa BarCode para filtrar...")
            with search_col2:
                if st.button("🔄 Ver Todos", key="clear_search_results"):
                    st.session_state['search_results_barcode'] = ""
                    st.rerun()

            
            # Results table - now supporting multiple types (PLACA1, PLACA2, SCADA1, etc.)
            results_data = []
            for result in st.session_state.batch_results:
                # Create one row per type (PLACA1, PLACA2, SCADA1, etc.)
                results_by_type = result.get('results_by_type', {})
                
                if results_by_type:
                    # Multiple types per barcode
                    for type_label, data in results_by_type.items():
                        results_data.append({
                            'BarCode': result['BarCode'],
                            'Tipo': type_label,
                            'Marca': data.get('marca', ''),
                            'Modelo': data.get('modelo', ''),
                            'Potencia': data.get('potencia', ''),
                            'Número de Serie': data.get('numero_serie', ''),
                            'Año': data.get('año', ''),
                            'Código SCADA': data.get('codigo_scada_principal', ''),
                            'Confianza': f"{data.get('confidence', 0):.2f}",
                            'Tiempo (s)': f"{result.get('processing_time', 0):.1f}",
                            'Estado': '✅' if result.get('status') == 'success' else '❌'
                        })
                else:
                    # Fallback for old format (shouldn't happen with new processing)
                    results_data.append({
                        'BarCode': result['BarCode'],
                        'Tipo': 'N/A',
                        'Marca': result.get('marca', ''),
                        'Modelo': result.get('modelo', ''),
                        'Potencia': '',
                        'Número de Serie': result.get('numero_serie', ''),
                        'Año': result.get('año', ''),
                        'Código SCADA': result.get('codigo_scada_principal', ''),
                        'Confianza': 'N/A',
                        'Tiempo (s)': f"{result.get('processing_time', 0):.1f}",
                        'Estado': '✅' if result.get('status') == 'success' else '❌'
                    })
            
            df_results = pd.DataFrame(results_data)
            
            # Apply search filter if specified
            if search_barcode and search_barcode.strip():
                df_results = df_results[df_results['BarCode'].str.contains(search_barcode.strip(), case=False, na=False)]
                st.caption(f"📊 Mostrando {len(df_results)} resultados que coinciden con '{search_barcode.strip()}'")
            
            st.dataframe(df_results, use_container_width=True, height=600)
            
            # Export button
            col_export1, col_export2 = st.columns([1, 1])
            
            with col_export1:
                generate_excel = st.button("💾 Generar Excel", key="export_unified", type="primary")
            
            logger.info(f"🔍 DEBUG: generate_excel button pressed = {generate_excel}")
            
            if generate_excel:
                logger.info("🚀 Starting Excel generation...")
                with st.spinner("⏳ Generando archivo Excel... (puede tardar unos segundos)"):
                    try:
                        logger.info("📊 Calling export_results()...")
                        export_results()
                        logger.info("✅ Excel generation completed successfully")
                        st.success("✅ Excel generado correctamente")
                    except Exception as e:
                        logger.error(f"❌ Error in export_results: {e}", exc_info=True)
                        st.error(f"❌ Error al generar Excel: {e}")
            
            # Show download button if Excel was generated
            if 'excel_export_data' in st.session_state:
                with col_export2:
                    st.download_button(
                        label="📥 Descargar Excel",
                        data=st.session_state.excel_export_data,
                        file_name=st.session_state.excel_export_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="secondary"
                    )
                
                st.caption(f"📊 **{st.session_state.excel_export_stats['rows']} filas** ({st.session_state.excel_export_stats['barcodes']} BarCodes únicos consolidados)")
                
                with st.expander("👁️ Vista Previa del Excel"):
                    st.dataframe(st.session_state.excel_export_df, use_container_width=True)
        else:
            if st.session_state.batch_data_df is None:
                st.info("ℹ️ No hay resultados aún. Carga un checkpoint usando el selector de archivos arriba")
            else:
                st.info("ℹ️ No hay resultados aún. Selecciona imágenes y procesa en la pestaña 'Selección'")


# Call main directly (Streamlit pages don't need if __name__ == "__main__")
main()
