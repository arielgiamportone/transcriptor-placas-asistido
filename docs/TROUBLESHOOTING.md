# 🔧 Troubleshooting - Problemas Comunes

## Instalación

### Error: "ModuleNotFoundError: No module named 'streamlit'"

**Solución:**
```bash
pip install -r requirements.txt
```

### Error: "CUDA not available" al usar OCR

**Causa:** No tienes GPU NVIDIA o drivers no instalados

**Solución:**
- ✅ Es NORMAL si no tienes GPU
- El OCR funciona con CPU (solo más lento)
- Para habilitar GPU:
  1. Instalar CUDA Toolkit 11.8+
  2. Instalar drivers NVIDIA actualizados
  3. Reinstalar PyTorch con soporte CUDA:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
     ```

### Error: "libGL.so.1: cannot open shared object file" (Linux)

**Solución:**
```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

---

## Uso de la Aplicación

### La aplicación no abre en el navegador

**Solución:**
1. Verifica que Streamlit esté corriendo:
   ```bash
   streamlit run app.py
   ```
2. Abre manualmente: `http://localhost:8501`
3. Si sigue sin funcionar, prueba otro puerto:
   ```bash
   streamlit run app.py --server.port 8502
   ```

### "Excel inválido o sin columna 'BarCode'"

**Causa:** Tu Excel no tiene la columna requerida

**Solución:**
1. Abre tu Excel
2. Verifica que exista una columna llamada exactamente `BarCode`
3. Si se llama "Codigo" o "ID", renómbrala a `BarCode`

### "No se encontraron columnas con imágenes"

**Causa:** Las columnas no contienen rutas a imágenes

**Solución:**
1. Las celdas deben contener rutas como:
   - `C:\Imagenes\motor1.jpg`
   - `../fotos/placa_18057.png`
2. Formatos soportados: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
3. Verifica que las rutas sean correctas y los archivos existan

### OCR muy lento (>30 segundos por imagen)

**Causa:** Primera vez carga modelos, o CPU lento

**Solución:**
- ✅ Primera imagen siempre es lenta (descarga modelos)
- ✅ Imágenes siguientes serán más rápidas (3-10s)
- Para acelerar:
  1. Usa GPU (ver arriba)
  2. O usa API OpenAI en lugar de OCR Local

### API OpenAI devuelve error 401

**Causa:** API Key incorrecta o no configurada

**Solución:**
1. Verifica tu API Key en https://platform.openai.com/api-keys
2. En Streamlit:
   - Sidebar → "⚙️ Configurar API Key"
   - Pega la key y guarda
3. O configura en `.env`:
   ```
   OPENAI_API_KEY=sk-tu-key-real-aqui
   ```

### API OpenAI devuelve error 429 (Rate Limit)

**Causa:** Excediste el límite de requests por minuto

**Solución:**
1. Espera 1-2 minutos
2. Procesa imágenes más lentamente
3. Actualiza tu plan de OpenAI: https://platform.openai.com/account/billing

### "Error al exportar Excel: No module named 'openpyxl'"

**Solución:**
```bash
pip install openpyxl
```

### Los campos no se llenan después de extraer

**Causa:** Error silencioso en extracción

**Solución:**
1. Abre la consola del navegador (F12)
2. Ve a la pestaña "Console"
3. Busca errores en rojo
4. Copia el error y repórtalo como issue en GitHub

---

## Problemas de Datos

### Los campos tienen valores incorrectos

**OCR Local:**
- Normal, el OCR tiene ~85% de precisión
- Revisa y corrige manualmente
- Considera usar API OpenAI para mayor precisión

**API OpenAI:**
- Poco común, pero puede ocurrir
- Reporta el caso con la imagen para mejorar el prompt

### La consolidación mezcla datos incorrectamente

**Ejemplo:** PLACA 1 tiene "ABB" pero PLACA 2 tiene "SIEMENS", y el resultado es "ABB + SIEMENS"

**Causa:** Ambas placas pertenecen al mismo motor pero son diferentes motores

**Solución:**
1. Verifica que todas las imágenes del mismo BarCode correspondan al mismo motor físico
2. Si son motores diferentes, usa BarCodes distintos

### Checkpoint no carga

**Error:** "Failed to load checkpoint"

**Solución:**
1. Verifica que el archivo `.json` no esté corrupto
2. Abre el JSON en un editor y verifica sintaxis
3. Si está corrupto, usa el checkpoint anterior:
   - Los checkpoints se nombran con timestamp
   - Busca el más reciente anterior al corrupto

---

## Rendimiento

### La aplicación consume mucha RAM (>8GB)

**Causa:** Modelos OCR + imágenes grandes en memoria

**Solución:**
1. Cierra otras aplicaciones
2. Procesa en lotes más pequeños (50-100 motores a la vez)
3. Reduce resolución de imágenes antes de procesarlas:
   ```bash
   # Ejemplo con ImageMagick
   mogrify -resize 50% *.jpg
   ```

### El navegador se congela al cargar Excel grande

**Causa:** Excel con >10,000 filas

**Solución:**
1. Divide el Excel en archivos más pequeños (1000 filas cada uno)
2. Procesa por partes
3. Al final, consolida los resultados con:
   ```python
   import pandas as pd
   df1 = pd.read_excel('resultados_parte1.xlsx')
   df2 = pd.read_excel('resultados_parte2.xlsx')
   df_total = pd.concat([df1, df2])
   df_total.to_excel('resultados_completos.xlsx', index=False)
   ```

---

## Deploy en Streamlit Cloud

### Error: "Package installation failed"

**Solución:**
1. Verifica `requirements.txt` no tenga versiones incompatibles
2. Usa versiones flexibles:
   ```
   pandas>=2.0.0,<3.0.0
   ```
3. Agrega `packages.txt` con dependencias del sistema:
   ```
   libgl1-mesa-glx
   libglib2.0-0
   ```

### App funciona local pero falla en Cloud

**Causa común:** Rutas absolutas en código

**Solución:**
Usa rutas relativas:
```python
# ❌ Mal
path = 'C:\\Users\\...\\Data'

# ✅ Bien
from pathlib import Path
path = Path(__file__).parent / 'Data'
```

### "Secrets not found" en Streamlit Cloud

**Solución:**
1. Ve a tu app en Streamlit Cloud
2. Settings → Secrets
3. Agrega:
   ```toml
   OPENAI_API_KEY = "sk-tu-key-aqui"
   ```

---

## Logs y Debugging

### ¿Dónde están los logs?

**Local:**
- `logs/app.log`
- Ver en tiempo real:
  ```bash
  tail -f logs/app.log
  ```

**Streamlit Cloud:**
- En el dashboard de la app → "Logs" (esquina inferior derecha)

### Habilitar modo debug

En `config.yaml`:
```yaml
logging:
  level: DEBUG  # Cambiar de INFO a DEBUG
```

Reiniciar la app.

---

## Contacto para Soporte

**No encuentras tu problema aquí?**

1. **GitHub Issues**: [github.com/tu-usuario/repo/issues](https://github.com)
2. **Email**: tu-email@ejemplo.com
3. **Discord**: [Unirse al servidor](https://discord.gg/...)

**Al reportar un problema, incluye:**
- Versión de Python (`python --version`)
- Sistema operativo
- Archivo `logs/app.log` (últimas 50 líneas)
- Pasos para reproducir el error
- Screenshot si es posible

---

**Última actualización:** Diciembre 2025
