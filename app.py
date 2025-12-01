"""
Aplicación Principal - Transcripción de Placas Industriales
Punto de entrada con navegación multi-página
"""

import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Transcripción Industrial",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏭 Sistema de Transcripción de Placas Industriales")
    st.markdown("---")
    
    st.markdown("""
    ### Bienvenido al Sistema de Transcripción
    
    Este sistema ofrece dos modalidades de trabajo integradas:
    
    #### 📝 Transcripción Asistida
    - Procesamiento **fila por fila** con revisión manual
    - Selección flexible de imágenes y métodos por activo
    - **Consolidación inteligente** de datos de múltiples imágenes
    - Visualización en tiempo real con respuestas de API
    - **Checkpoints automáticos** para no perder progreso
    - **Puede revisar resultados del procesamiento batch**
    
    #### ⚡ Procesamiento Rápido en Batch
    - Procesamiento **masivo** de múltiples activos
    - Selección visual rápida con miniaturas
    - Procesa todas las imágenes seleccionadas de una vez
    - Exportación directa a Excel
    - **Los resultados pueden revisarse luego en Transcripción Asistida**
    
    ---
    
    ### � Flujo de Trabajo Integrado
    
    1. **⚡ Procesamiento Rápido** → Procesa 50-100 activos en batch (rápido)
    2. **💾 Exporta** los resultados a Excel
    3. **📝 Transcripción Asistida** → Carga el mismo Excel y revisa/corrige
    4. Los **checkpoints** permiten retomar donde lo dejaste
    
    O simplemente usa **📝 Transcripción Asistida** desde el inicio si prefieres máximo control.
    
    ---
    
    ### � Selecciona una página en la barra lateral para comenzar
    """)
    
    # Quick comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📝 Transcripción Asistida**
        
        ✅ Máxima precisión  
        ✅ Control total  
        ✅ Revisión inmediata  
        ✅ Consolidación multi-imagen  
        ✅ Ver respuesta de API  
        ✅ Puede cargar resultados batch  
        
        ⏱️ Tiempo: ~2-3 min/activo
        """)
    
    with col2:
        st.success("""
        **⚡ Procesamiento Rápido**
        
        ✅ Alta velocidad  
        ✅ Procesamiento masivo  
        ✅ Exportación directa  
        ✅ Revisión posterior  
        ✅ Selección visual rápida  
        ✅ Compatible con Transcripción  
        
        ⏱️ Tiempo: ~5-10 seg/activo
        """)
    
    st.markdown("---")
    
    # Configuration section
    with st.expander("⚙️ Configuración y Ayuda"):
        st.markdown("""
        ### 💡 Cómo usar este sistema
        
        **Primer Uso:**
        1. Ve a **📝 Transcripción Asistida** (barra lateral)
        2. Configura tu API Key de OpenAI
        3. Carga tu archivo Excel
        
        **Procesamiento Rápido:**
        1. Ve a **⚡ Procesamiento Rápido** (barra lateral)
        2. Carga el mismo Excel
        3. Selecciona imágenes masivamente
        4. Procesa todo en batch
        5. Exporta resultados
        
        **Revisión de Resultados:**
        1. Vuelve a **📝 Transcripción Asistida**
        2. Carga el Excel (o continúa donde lo dejaste)
        3. Los checkpoints guardan tu progreso automáticamente
        
        ---
        
        ### API Configuration
        
        La API Key se configura en **Transcripción Asistida** y se comparte automáticamente con **Procesamiento Rápido**.
        
        ### Modelos Disponibles
        - **gpt-4o-mini** (Recomendado): Rápido y económico (~$0.15 por 1000 imágenes)
        - **gpt-4o**: Mayor precisión (~$2.50 por 1000 imágenes)
        - **gpt-4-turbo**: Análisis profundo (~$10 por 1000 imágenes)
        
        ### Estructura del Excel
        - Columna **BarCode**: Identificador único del activo
        - Columnas desde **BJ** en adelante: URLs o paths de imágenes
        
        ### Estado Compartido
        
        Ambas páginas pueden trabajar con el mismo Excel. Los checkpoints de Transcripción Asistida
        son independientes, pero puedes cargar los resultados del batch y revisarlos uno por uno.
        """)


if __name__ == "__main__":
    main()
