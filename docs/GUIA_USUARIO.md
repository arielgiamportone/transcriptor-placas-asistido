# 📚 Guía de Usuario - Transcriptor de Placas Asistido V1

## Índice
- [Inicio Rápido](#inicio-rápido)
- [Flujo de Trabajo Completo](#flujo-de-trabajo-completo)
- [Funcionalidades Avanzadas](#funcionalidades-avanzadas)
- [Preguntas Frecuentes](#preguntas-frecuentes)

---

## Inicio Rápido

### 1. Preparar tus Datos

Tu archivo Excel debe tener esta estructura:

```
| BarCode | Imagen1 | Imagen2 | Imagen3 |
|---------|---------|---------|---------|
| 18057   | path/to/img1.jpg | path/to/img2.jpg | path/to/img3.jpg |
| 18058   | path/to/img4.jpg | path/to/img5.jpg | |
```

**Importante:**
- Columna `BarCode` es obligatoria
- Las rutas pueden ser absolutas o relativas al Excel
- Formatos soportados: JPG, PNG, BMP, TIFF

### 2. Ejecutar la Aplicación

```bash
streamlit run app.py
```

Se abrirá automáticamente en tu navegador: `http://localhost:8501`

### 3. Cargar Excel

1. En la barra lateral, clic en **"📁 Cargar Excel"**
2. Selecciona tu archivo `.xlsx` o `.xls`
3. Espera unos segundos mientras se carga

**Verás:**
- Total de BarCodes encontrados
- Total de imágenes detectadas
- Vista previa de la primera fila

---

## Flujo de Trabajo Completo

### Paso 1: Seleccionar BarCode

En la parte superior, verás:
```
BarCode: [18057 ▼]  [< Anterior] [Siguiente >]
```

- Usa el dropdown para saltar a cualquier BarCode
- O navega secuencialmente con las flechas

### Paso 2: Ver Imágenes

Se mostrarán dos vistas:
- **Original**: Imagen tal como está en el archivo
- **Preprocesada**: Imagen mejorada automáticamente para OCR

**Controles:**
- Zoom con mouse wheel
- Pan con clic y arrastre
- Click en "🔍 Ver en Canvas OCR" para marcar zonas

### Paso 3: Seleccionar Tipo de Imagen

Dropdown **"Tipo de Imagen"**:
- **🔄 AMBOS**: Contiene placa técnica Y código SCADA
- **📋 PLACA 1**: Primera placa técnica del motor
- **📋📋 PLACA 2**: Segunda placa (si hay múltiples)
- **📋📋📋 PLACA 3**: Tercera placa
- **🔢 SCADA 1**: Primer código SCADA
- **🔢🔢 SCADA 2**: Segundo código SCADA
- **🔢🔢🔢 SCADA 3**: Tercer código SCADA
- **❌ Omitir**: Imagen irrelevante, saltarla

**Ejemplo de uso:**
- Motor con 1 placa + 1 SCADA → Procesar imagen 1 como "PLACA 1", imagen 2 como "SCADA 1"
- Motor con 2 placas → Imagen 1 como "PLACA 1", imagen 2 como "PLACA 2"

### Paso 4: Elegir Método de Extracción

#### Opción A: OCR Local (Gratis)

1. Seleccionar **"🤖 OCR Local"** en el dropdown
2. Click **"🔍 Extraer con OCR"**
3. Espera 5-15 segundos (primera vez más lento)
4. Verás los campos llenados automáticamente

**Ventajas:**
- ✅ Gratis, ilimitado
- ✅ Funciona offline
- ✅ Datos no salen de tu computadora

**Desventajas:**
- ⚠️ Más lento (~10s por imagen)
- ⚠️ Menos preciso (~85% accuracy)
- ⚠️ Requiere más corrección manual

#### Opción B: API OpenAI (Pago)

1. Configurar API Key (solo primera vez):
   - Sidebar → **"⚙️ Configurar API Key"**
   - Pegar tu key de OpenAI
   - Click "Guardar"

2. Seleccionar **"🌐 API OpenAI"** en dropdown
3. Elegir modelo:
   - **GPT-4o-mini** (recomendado): Rápido y económico
   - **GPT-4o**: Más preciso, más caro
   - **GPT-4 Turbo**: Máxima precisión, máximo costo

4. Click **"🤖 Extraer con API"**
5. Espera 3-8 segundos
6. Campos llenados con alta precisión

**Ventajas:**
- ✅ Muy rápido (~5s por imagen)
- ✅ Alta precisión (~95% accuracy)
- ✅ Requiere mínima corrección

**Desventajas:**
- ⚠️ Costo: ~$0.002 por imagen (GPT-4o-mini)
- ⚠️ Requiere conexión a internet
- ⚠️ Datos enviados a OpenAI (encriptados)

### Paso 5: Revisar y Corregir Campos

Los campos aparecerán con colores:
- 🟢 **Verde**: Alta confianza (probablemente correcto)
- 🟡 **Amarillo**: Confianza media (revisar)
- 🔴 **Rojo**: Error o campo vacío (requiere atención)

**Campos disponibles:**
- Marca (ej: ABB, SIEMENS, WEG)
- Modelo (ej: M2BAX 100LA4)
- Número de Serie
- Año de fabricación
- Potencia (ej: 3 kW)
- Voltaje (ej: 380V)
- Corriente (ej: 6.5A)
- Frecuencia (ej: 50 Hz)
- RPM (ej: 1500)
- Factor de Potencia (ej: 0.85)
- Eficiencia (ej: IE3)
- IP (ej: IP55)
- Clase de Aislamiento (ej: F)
- Tipo de Motor (ej: Trifásico)
- Conexión (ej: Y/Δ)
- Rodamiento DE/NDE
- Código SCADA Principal
- Código SCADA Respaldo
- Código TAG

**Atajos de teclado:**
- `Tab`: Siguiente campo
- `Enter`: (en último campo) Guardar
- `Ctrl + S`: Guardar en cualquier momento

### Paso 6: Guardar y Continuar

Click **"💾 Guardar y Continuar"**

**Lo que sucede:**
1. Datos se consolidan con imágenes anteriores del mismo BarCode
2. Se crea checkpoint automático
3. Avanza a la siguiente imagen automáticamente

**Consolidación inteligente:**
Si en PLACA 1 tienes `Marca: ABB` y en PLACA 2 tienes `Marca: ABB`, el resultado final será `Marca: ABB` (sin duplicar).

Si en PLACA 1 tienes `Potencia: 3 kW` y en PLACA 2 tienes `Potencia: 5 kW`, el resultado será `Potencia: 3 kW + 5 kW`.

### Paso 7: Exportar Resultados

Cuando hayas terminado:

1. Ve al tab **"📊 Resultados"**
2. Verás tabla con todos los activos procesados
3. Click **"📊 Exportar Excel Consolidado"**

Se creará archivo: `outputs/transcription_consolidated.xlsx`

**Formato del Excel:**
- Una fila por BarCode
- Todos los datos de PLACA 1, 2, 3 fusionados
- Todos los códigos SCADA concatenados

---

## Funcionalidades Avanzadas

### Canvas OCR Interactivo

1. Click **"🔍 Ver en Canvas OCR"** junto a la imagen
2. Se abre ventana con herramientas:
   - **Dibujar rectángulo**: Marca zona de interés
   - **Zoom**: Acerca/aleja la imagen
   - **Borrar**: Limpia marcas

3. Marca la zona con texto relevante
4. Click **"📋 Copiar Texto OCR"**
5. Texto se copia al portapapeles
6. Pégalo en el campo correspondiente

**Útil cuando:**
- OCR falló en un campo específico
- Necesitas copiar un número de serie complejo
- Quieres aislar una sección de la placa

### Checkpoints Automáticos

El sistema guarda automáticamente después de cada imagen procesada.

**Beneficios:**
- Nunca pierdes tu progreso
- Puedes cerrar y continuar después
- Si crashea, recuperas el trabajo

**Para continuar:**
1. Sidebar → **"📂 Continuar desde Checkpoint"**
2. Selecciona el checkpoint más reciente
3. Click **"Cargar"**
4. Continúa donde lo dejaste

**Ubicación:** `outputs/checkpoints/checkpoint_<timestamp>.json`

### Validación Cruzada

El sistema valida automáticamente:

1. **Marca vs Modelo**: Verifica que el modelo corresponde a la marca
   - ABB → Modelos M2, M3, M4
   - SIEMENS → Modelos 1LA, 1LE, 1LG
   - WEG → Modelos W22, W21

2. **P = V × I**: Verifica cálculo de potencia
   - Tolerancia: ±20%
   - Ejemplo: 3 kW ≈ 380V × 6.5A

3. **Año válido**: Entre 1950 y 2030

**Si hay advertencias:**
- Se muestra mensaje amarillo en la UI
- Revisa manualmente los campos marcados
- Corrige si es necesario

---

## Preguntas Frecuentes

### ¿Cuánto cuesta usar la API de OpenAI?

**GPT-4o-mini** (recomendado):
- Costo: ~$0.002 por imagen
- 1000 imágenes ≈ $2
- Suficiente para la mayoría de casos

**GPT-4o**:
- Costo: ~$0.01 por imagen
- Más preciso, pero 5x más caro

### ¿Funciona sin conexión a internet?

**Sí**, con OCR Local:
- Descargas los modelos una vez
- Después funciona 100% offline
- Sin costo

**No**, con API OpenAI:
- Requiere conexión para llamar la API
- Alternativa: OCR Local

### ¿Qué tan preciso es el OCR?

**OCR Local:**
- Placas nítidas, bien iluminadas: ~90%
- Placas desgastadas, borrosas: ~70%
- Requiere corrección manual

**API OpenAI (GPT-4o-mini):**
- Placas nítidas: ~95%
- Placas desgastadas: ~85%
- Muy resistente a calidad baja

### ¿Puedo procesar varias imágenes a la vez?

Actualmente no (modo Transcripción Asistida).

Para procesamiento batch, ver:
- Tab **"⚡ Procesamiento Rápido"** (próximamente)

### ¿Cómo manejo motores con múltiples placas?

Ejemplo: Motor con 2 placas técnicas y 1 código SCADA

**Paso 1:** Imagen 1 → Tipo: **PLACA 1** → Extraer → Guardar  
**Paso 2:** Imagen 2 → Tipo: **PLACA 2** → Extraer → Guardar  
**Paso 3:** Imagen 3 → Tipo: **SCADA 1** → Extraer → Guardar

Al exportar, todos los datos se consolidan en una sola fila:
```
BarCode | Marca | Modelo (de PLACA 1 + PLACA 2) | ... | SCADA (de SCADA 1)
18057   | ABB   | M2BAX 100LA4 + M2BAX 100LA4   | ... | 6MT-1234
```

### ¿Se pueden editar resultados ya guardados?

**Sí**:
1. Navega al BarCode que quieres editar
2. Los campos se llenarán con los datos guardados
3. Modifica lo que necesites
4. Click **"💾 Guardar"**
5. Se sobrescribe el checkpoint

### ¿Los checkpoints ocupan mucho espacio?

No. Cada checkpoint es un archivo JSON de ~5-20 KB.

1000 checkpoints ≈ 10 MB

**Limpieza automática:**
El sistema mantiene solo los últimos 50 checkpoints.

### ¿Cómo exporto solo ciertos BarCodes?

En el tab **"📊 Resultados"**:

1. Usa el buscador: **"🔍 Buscar por BarCode"**
2. Filtra los que necesitas
3. Click **"📊 Exportar Excel Consolidado"**
4. El Excel contendrá solo los filtrados

### ¿Puedo usar Google Lens u otro OCR?

Actualmente no integrado.

**Alternativa:**
1. Usa Google Lens externamente
2. Copia el texto
3. Pégalo manualmente en los campos

Planeamos integrar más motores OCR en V2.0.

---

## Soporte

**Problemas comunes:**
- Ver [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**Reportar bugs:**
- GitHub Issues: [github.com/tu-usuario/repo/issues]

**Contacto:**
- Email: tu-email@ejemplo.com

---

¡Gracias por usar el Transcriptor de Placas Asistido! 🚀
