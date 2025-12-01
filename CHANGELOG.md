# Changelog

Todos los cambios notables en este proyecto serán documentados aquí.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

---

## [1.0.0] - 2025-12-01

### 🎉 Lanzamiento Inicial

Primera versión estable del **Transcriptor de Placas Industriales Asistido**.

### ✨ Características Agregadas

#### Core
- Sistema de transcripción asistida con validación humana fila por fila
- Soporte dual: OCR Local (EasyOCR) + API OpenAI (GPT-4o-mini/4o/4o-turbo)
- Consolidación inteligente de múltiples imágenes por BarCode
- Checkpoints automáticos después de cada imagen procesada
- Validación cruzada automática (marca/modelo, P=V×I, años)

#### UI/UX
- Interfaz web con Streamlit (modo multi-página)
- Tab "Transcripción Asistida" con procesamiento on-demand
- Tab "Resultados" con vista consolidada de todos los activos
- Canvas OCR interactivo para marcar zonas de interés
- Código de colores por confianza (verde/amarillo/rojo)
- Atajos de teclado (Tab, Enter, flechas)

#### Tipos de Imágenes
- **AMBOS**: Placa técnica + código SCADA en una imagen
- **PLACA 1/2/3**: Múltiples placas técnicas por motor
- **SCADA 1/2/3**: Múltiples códigos SCADA
- **Omitir**: Saltar imágenes irrelevantes

#### Extracción de Datos
- 20+ campos soportados:
  - Placa Técnica: Marca, Modelo, N° Serie, Año, Potencia, Voltaje, Corriente, Frecuencia, RPM, FP, Eficiencia, IP, Clase Aislamiento, Tipo Motor, Conexión, Rodamientos
  - Código SCADA: Principal, Respaldo, TAG
- Preprocesamiento automático de imágenes (deskew, denoise, contrast)
- Formato de salida estandarizado con unidades ("kW", "V", "Hz", "rpm")

#### Exportación
- **Excel Consolidado** (.xlsx): Una fila por BarCode con todos los datos fusionados ⭐
- CSV: Tabla expandida con todas las extracciones
- JSON: Formato raw con metadata completa

#### Validación
- Marca vs Modelo: Verifica patrones conocidos (ABB→M2/M3/M4, SIEMENS→1LA/1LE/1LG)
- Cálculo de potencia: P = V × I (tolerancia ±20%)
- Año válido: Rango 1950-2030
- Normalización de formatos automática

#### Configuración
- Archivo `config.yaml` centralizado
- Variables de entorno con `.env` para API keys
- Configuración de API desde UI (sidebar)

#### Documentación
- README.md profesional con badges y ejemplos
- Guía de usuario detallada (`docs/GUIA_USUARIO.md`)
- Troubleshooting exhaustivo (`docs/TROUBLESHOOTING.md`)
- Licencia MIT

#### Deploy
- Dockerfile para contenedores
- Configuración para Streamlit Cloud (`packages.txt`, `.streamlit/config.toml`)
- `.gitignore` completo
- Estructura de repositorio lista para GitHub

### 🔧 Configuración

- Python 3.10+ requerido
- Dependencias especificadas en `requirements.txt` con versionado semántico
- Soporte para GPU opcional (CUDA) para acelerar OCR

### 📊 Métricas de Rendimiento

**OCR Local (EasyOCR):**
- Velocidad: ~10-15s por imagen (CPU) / ~3-5s (GPU)
- Precisión: ~85-90%
- Costo: $0 (gratis)

**API OpenAI (GPT-4o-mini):**
- Velocidad: ~3-8s por imagen
- Precisión: ~95-97%
- Costo: ~$0.002 por imagen

### 🐛 Problemas Conocidos

- OCR Local puede fallar con placas muy desgastadas o borrosas (usar API como alternativa)
- Primera ejecución de OCR es lenta (~30s) debido a descarga de modelos
- Consolidación puede mezclar datos si múltiples motores usan el mismo BarCode

### 🔐 Seguridad

- API keys almacenadas en `.env` (no versionadas en Git)
- Secrets de Streamlit Cloud soportados
- Comunicación HTTPS con APIs de terceros

---

## [Unreleased] - Próximas Versiones

### 🚧 En Desarrollo

#### V1.1 (Q1 2026)
- [ ] Modo "Procesamiento Rápido" (batch) con configuración por imagen
- [ ] Dashboard de estadísticas avanzadas
- [ ] Exportación a XML y otros formatos
- [ ] Importación desde resultados de batch
- [ ] Atajos de teclado personalizables
- [ ] Modo oscuro

#### V2.0 (Q2 2026)
- [ ] Fine-tuning de modelos con datos propios
- [ ] Detección automática de método óptimo (OCR vs API) por imagen
- [ ] Sistema multi-usuario con roles y permisos
- [ ] API REST para integración con ERP/CMMS
- [ ] Mobile app (iOS/Android) para captura en campo

### 💡 Ideas Bajo Consideración

- Google Lens integration
- Tesseract OCR como motor alternativo
- Soporte para placas en alemán, chino, portugués
- Exportación directa a SAP, Maximo, otros CMMS
- Blockchain para auditoría inmutable
- Sistema de templates personalizables por industria

---

## Convenciones de Versionado

- **MAJOR** (X.0.0): Cambios incompatibles con versiones anteriores
- **MINOR** (1.X.0): Nueva funcionalidad compatible hacia atrás
- **PATCH** (1.0.X): Correcciones de bugs compatibles

---

**Última actualización:** 1 de Diciembre de 2025
