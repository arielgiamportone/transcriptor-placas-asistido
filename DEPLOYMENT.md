# 🚀 Guía de Despliegue

Esta guía detalla los pasos para subir el proyecto a GitHub y desplegarlo en Streamlit Cloud.

---

## 📋 Requisitos Previos

- Cuenta en [GitHub](https://github.com)
- Cuenta en [Streamlit Cloud](https://streamlit.io/cloud) (gratis)
- API Key de OpenAI (si usarás API en producción)

---

## 1️⃣ Subir a GitHub

### Paso 1: Crear Repositorio en GitHub

1. Ve a https://github.com/new
2. Configuración:
   - **Repository name**: `transcriptor-placas-asistido` (o el nombre que prefieras)
   - **Description**: "🏭 Sistema inteligente de transcripción asistida de placas técnicas industriales con OCR + GPT-4o"
   - **Visibility**: 
     - ✅ **Public** (recomendado para Streamlit Cloud gratuito)
     - 🔒 **Private** (requiere Streamlit Cloud Pro)
   - **NO** marcar "Initialize with README" (ya tenemos uno)
3. Click "Create repository"

### Paso 2: Conectar Repositorio Local

GitHub te mostrará comandos. Usa estos:

```bash
cd "C:\Users\Ariel Giamporte\Desktop\appPlacas\Transcriptor de placas Asistido V1"

# Agregar el remote
git remote add origin https://github.com/TU-USUARIO/transcriptor-placas-asistido.git

# Renombrar rama a main (estándar de GitHub)
git branch -M main

# Push inicial
git push -u origin main
```

### Paso 3: Verificar

1. Actualiza la página de tu repositorio en GitHub
2. Deberías ver:
   - ✅ README.md renderizado con badges
   - ✅ 25 archivos
   - ✅ Estructura de carpetas (docs/, pages/, etc.)
   - ✅ LICENSE visible

---

## 2️⃣ Desplegar en Streamlit Cloud

### Paso 1: Conectar Streamlit con GitHub

1. Ve a https://share.streamlit.io/
2. Click **"New app"**
3. Autoriza a Streamlit a acceder a tu cuenta de GitHub (si es primera vez)

### Paso 2: Configurar App

En el formulario:

1. **Repository**: Selecciona `TU-USUARIO/transcriptor-placas-asistido`
2. **Branch**: `main`
3. **Main file path**: `app.py`
4. **App URL** (opcional): Personaliza la URL
   - Ejemplo: `transcriptor-placas.streamlit.app`

### Paso 3: Configurar Secrets

**MUY IMPORTANTE**: Antes de hacer deploy, agrega tu API Key:

1. Click en **"Advanced settings"** (abajo del formulario)
2. En la sección **"Secrets"**, pega esto:

```toml
OPENAI_API_KEY = "sk-tu-api-key-real-aqui"

# Opcional: Otras APIs
# ANTHROPIC_API_KEY = "sk-ant-..."
# GOOGLE_API_KEY = "AIza..."
```

3. **NO** uses comillas simples, solo dobles
4. **NO** compartas esta key públicamente

### Paso 4: Deploy

1. Click **"Deploy!"**
2. Espera 2-5 minutos mientras instala dependencias
3. Verás logs en tiempo real:
   ```
   [manager] Installing dependencies...
   [manager] Collecting streamlit...
   [manager] Installing collected packages...
   [manager] Successfully installed...
   ```

### Paso 5: Verificar

Si todo va bien:
- ✅ App se abre automáticamente
- ✅ Sidebar muestra "Transcriptor de Placas Industriales"
- ✅ Puedes navegar a "📝 Transcripción Asistida"

---

## 3️⃣ Troubleshooting del Deploy

### Error: "Package installation failed"

**Causa**: Dependencia incompatible en `requirements.txt`

**Solución**:
1. Revisa logs para ver qué paquete falló
2. En tu repositorio local:
   ```bash
   # Ejemplo: Si falla opencv-python-headless
   # Edita requirements.txt, cambia versión a:
   opencv-python-headless>=4.8.0,<5.0.0
   
   git add requirements.txt
   git commit -m "fix: ajustar versión de opencv"
   git push
   ```
3. Streamlit Cloud re-deployará automáticamente

### Error: "App is in an unhealthy state"

**Causa**: App crashea al iniciar

**Solución**:
1. Ve a Streamlit Cloud → Tu app → "Manage app" → "Logs"
2. Busca el error exacto (última línea roja)
3. Comunes:
   - **"No module named 'X'"**: Falta en requirements.txt → Agrégalo
   - **"FileNotFoundError"**: Ruta absoluta en código → Cambia a relativa
   - **"OPENAI_API_KEY not found"**: Revisa secrets (Paso 3)

### Error: "Your app has exceeded the resource limits"

**Causa**: App consume mucha RAM (>1GB en plan gratuito)

**Solución**:
- Modelos OCR grandes no funcionarán en plan gratuito
- **Recomendación**: Usar solo API OpenAI en producción (más ligero)
- O contratar Streamlit Cloud Pro ($20/mes, 7GB RAM)

### Error en logs: "libGL.so.1: cannot open shared object file"

**Causa**: Faltan librerías del sistema para OpenCV

**Solución**:
- Ya está cubierto en `packages.txt`
- Si sigue fallando, agrega más paquetes en `packages.txt`:
  ```
  libgl1-mesa-glx
  libglib2.0-0
  libsm6
  libxext6
  libxrender-dev
  libgomp1
  libgfortran5  # Agregar esta línea
  ```
- Commit y push:
  ```bash
  git add packages.txt
  git commit -m "fix: agregar libgfortran5 para OpenCV"
  git push
  ```

---

## 4️⃣ Configuración Post-Deploy

### Actualizar Secrets

Si necesitas cambiar la API Key después del deploy:

1. Streamlit Cloud → Tu app → Hamburger menu (☰) → "Settings"
2. Pestaña "Secrets"
3. Edita el contenido
4. Click "Save"
5. App reiniciará automáticamente

### Personalizar URL

1. Settings → "General"
2. En "App URL", cambia el nombre
3. Ejemplo: `transcriptor-placas-ariel.streamlit.app`

### Configurar Branch de Deploy

Si trabajas en una rama de desarrollo:

1. Settings → "General"
2. Cambia "Branch" a `develop` o `staging`
3. Tu app se actualizará con cada push a esa rama

---

## 5️⃣ Workflow de Desarrollo

### Desarrollo Local

```bash
# 1. Crear rama para nueva feature
git checkout -b feature/nueva-funcionalidad

# 2. Hacer cambios...

# 3. Probar localmente
streamlit run app.py

# 4. Commit
git add .
git commit -m "feat: agregar nueva funcionalidad"

# 5. Push
git push origin feature/nueva-funcionalidad
```

### Pull Request y Merge

1. Ve a GitHub → Tu repo → "Pull requests" → "New"
2. Compara `feature/nueva-funcionalidad` → `main`
3. Crea PR, describe cambios
4. Revisa, aprueba, merge
5. Streamlit Cloud detectará cambios en `main` y re-deployará automáticamente

### Rollback si Algo Sale Mal

```bash
# Ver commits recientes
git log --oneline

# Revertir al commit anterior
git revert HEAD

# Push del revert
git push

# Streamlit Cloud volverá a la versión anterior
```

---

## 6️⃣ Monitoreo

### Métricas de la App

Streamlit Cloud provee:
- **Viewer count**: Usuarios activos
- **Usage**: RAM, CPU
- **Uptime**: Disponibilidad

Accede en: Dashboard → Tu app → "Analytics"

### Logs en Vivo

Para debuggear en producción:
1. Dashboard → Tu app → "Manage app"
2. Click "View logs" (botón abajo a la derecha)
3. Verás logs en tiempo real

---

## 7️⃣ Deploy con Docker (Opcional)

Si prefieres deployar en tu propio servidor:

### Build

```bash
cd "C:\Users\Ariel Giamporte\Desktop\appPlacas\Transcriptor de placas Asistido V1"

docker build -t transcriptor-placas:latest .
```

### Run

```bash
docker run -d \
  -p 8501:8501 \
  -e OPENAI_API_KEY="sk-tu-key-aqui" \
  -v $(pwd)/Data:/app/Data \
  -v $(pwd)/outputs:/app/outputs \
  --name transcriptor-placas \
  transcriptor-placas:latest
```

Abre: http://localhost:8501

### Docker Compose (más fácil)

Crea `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./Data:/app/Data
      - ./outputs:/app/outputs
      - ./logs:/app/logs
    restart: unless-stopped
```

Luego:

```bash
# Crear .env con tu key
echo "OPENAI_API_KEY=sk-..." > .env

# Correr
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener
docker-compose down
```

---

## 8️⃣ Checklist Final

Antes de considerar el deploy completo:

- [ ] ✅ Repositorio público en GitHub
- [ ] ✅ README visible y formateado
- [ ] ✅ LICENSE presente
- [ ] ✅ App funcionando en Streamlit Cloud
- [ ] ✅ Secrets configurados (API Key)
- [ ] ✅ URL personalizada (opcional)
- [ ] ✅ Probaste cargar un Excel de ejemplo
- [ ] ✅ Probaste la transcripción asistida (OCR o API)
- [ ] ✅ Exportación a Excel consolidado funciona
- [ ] ✅ Logs no muestran errores críticos

---

## 📞 Soporte

**Problemas durante el deploy?**

1. **GitHub Issues**: [tu-repo/issues](https://github.com/tu-usuario/transcriptor-placas-asistido/issues)
2. **Streamlit Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
3. **Email**: tu-email@ejemplo.com

---

**¡Felicidades! Tu app está en producción 🎉**
