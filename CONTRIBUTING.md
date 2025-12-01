# 🤝 Guía de Contribución

¡Gracias por tu interés en contribuir al **Transcriptor de Placas Industriales Asistido**! 

Este documento describe cómo puedes ayudar a mejorar el proyecto.

---

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [¿Cómo Puedo Contribuir?](#cómo-puedo-contribuir)
- [Configuración del Entorno de Desarrollo](#configuración-del-entorno-de-desarrollo)
- [Proceso de Desarrollo](#proceso-de-desarrollo)
- [Estándares de Código](#estándares-de-código)
- [Proceso de Pull Request](#proceso-de-pull-request)
- [Reportar Bugs](#reportar-bugs)
- [Sugerir Mejoras](#sugerir-mejoras)

---

## 📜 Código de Conducta

Este proyecto adhiere a un Código de Conducta basado en respeto y colaboración:

- **Se respetuoso** con otros contribuidores
- **Se constructivo** en tus comentarios
- **Se paciente** con principiantes
- **Prohibido**: Acoso, discriminación, lenguaje ofensivo

---

## 🚀 ¿Cómo Puedo Contribuir?

### Formas de Contribuir

1. **Reportar bugs** → [Ver sección](#reportar-bugs)
2. **Sugerir funcionalidades** → [Ver sección](#sugerir-mejoras)
3. **Mejorar documentación**
4. **Escribir código**
5. **Revisar Pull Requests**
6. **Responder preguntas** en Issues/Discussions

### Áreas que Necesitan Ayuda

- ✅ **Testing**: Más tests unitarios e integración
- 📚 **Documentación**: Tutoriales, videos, ejemplos
- 🌐 **Internacionalización**: Traducción a otros idiomas
- 🎨 **UI/UX**: Mejoras en interfaz
- 🤖 **Modelos**: Fine-tuning, nuevos motores OCR
- 🔌 **Integraciones**: SAP, Maximo, otros CMMS

---

## ⚙️ Configuración del Entorno de Desarrollo

### Requisitos

- Python 3.10 o superior
- Git
- Cuenta de GitHub

### Instalación

```bash
# 1. Fork el repositorio en GitHub
# (Click en "Fork" en la esquina superior derecha)

# 2. Clonar tu fork
git clone https://github.com/TU-USUARIO/transcriptor-placas-asistido.git
cd transcriptor-placas-asistido

# 3. Agregar el repositorio original como remote
git remote add upstream https://github.com/USUARIO-ORIGINAL/transcriptor-placas-asistido.git

# 4. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows

# 5. Instalar dependencias + herramientas de desarrollo
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Si existe

# 6. Instalar pre-commit hooks (opcional pero recomendado)
pre-commit install

# 7. Verificar instalación
pytest tests/
```

---

## 🔄 Proceso de Desarrollo

### 1. Crear una Rama

**Importante:** Nunca trabajes directamente en `main`

```bash
# Actualizar main
git checkout main
git pull upstream main

# Crear nueva rama
git checkout -b feature/nombre-descriptivo
# o
git checkout -b fix/bug-descripcion
```

**Convención de nombres:**
- `feature/` → Nueva funcionalidad
- `fix/` → Corrección de bug
- `docs/` → Cambios en documentación
- `refactor/` → Refactorización sin cambiar funcionalidad
- `test/` → Agregar/mejorar tests

### 2. Hacer Cambios

```bash
# Editar archivos...

# Ver cambios
git status
git diff

# Agregar cambios
git add archivo1.py archivo2.py

# Commit con mensaje descriptivo
git commit -m "feat: agregar soporte para Google Lens OCR"
```

### 3. Convención de Commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/es/):

```
<tipo>[alcance opcional]: <descripción>

[cuerpo opcional]

[footer(s) opcional(es)]
```

**Tipos:**
- `feat`: Nueva funcionalidad
- `fix`: Corrección de bug
- `docs`: Cambios en documentación
- `style`: Formato (sin cambiar código)
- `refactor`: Refactorización
- `test`: Agregar/actualizar tests
- `chore`: Tareas de mantenimiento

**Ejemplos:**
```bash
git commit -m "feat(ocr): agregar soporte para Tesseract"
git commit -m "fix(api): corregir timeout en llamadas a OpenAI"
git commit -m "docs: actualizar README con ejemplos de Docker"
```

### 4. Mantener tu Rama Actualizada

```bash
# Actualizar main local
git checkout main
git pull upstream main

# Volver a tu rama y hacer rebase
git checkout feature/tu-rama
git rebase main

# Si hay conflictos, resuélvelos y:
git add .
git rebase --continue
```

### 5. Push a tu Fork

```bash
git push origin feature/tu-rama
```

---

## 🎨 Estándares de Código

### Python Style Guide

Seguimos [PEP 8](https://peps.python.org/pep-0008/)

**Herramientas:**
```bash
# Formatear código automáticamente
black .

# Ordenar imports
isort .

# Linter
flake8 .
pylint src/

# Type checking
mypy src/
```

### Reglas Específicas

1. **Nombres:**
   - Variables: `snake_case`
   - Clases: `PascalCase`
   - Constantes: `UPPER_CASE`
   - Funciones: `snake_case`

2. **Docstrings:**
   ```python
   def extract_data(image_path: str, model: str) -> dict:
       """
       Extrae datos de una imagen usando el modelo especificado.
       
       Args:
           image_path: Ruta a la imagen a procesar
           model: Nombre del modelo ("ocr", "gpt4o-mini", etc.)
       
       Returns:
           Diccionario con los datos extraídos
       
       Raises:
           FileNotFoundError: Si la imagen no existe
           ValueError: Si el modelo no es válido
       """
       pass
   ```

3. **Type Hints:**
   ```python
   # ✅ Bien
   def process_image(path: Path, config: dict) -> Optional[dict]:
       pass
   
   # ❌ Mal
   def process_image(path, config):
       pass
   ```

4. **Imports:**
   ```python
   # Orden:
   # 1. Standard library
   import os
   from pathlib import Path
   
   # 2. Third-party
   import pandas as pd
   import streamlit as st
   
   # 3. Local
   from api_extractor import APIExtractor
   from config import get_config
   ```

### Testing

**Todos los nuevos features deben incluir tests.**

```bash
# Ejecutar tests
pytest

# Con coverage
pytest --cov=src --cov-report=html

# Solo un test específico
pytest tests/test_api_extractor.py::test_gpt4o_extraction
```

**Ejemplo de test:**
```python
import pytest
from api_extractor import APIExtractor

def test_extract_with_valid_image():
    """Test extracción exitosa con imagen válida"""
    extractor = APIExtractor(model="gpt-4o-mini")
    result = extractor.extract("tests/fixtures/motor_placa.jpg", "placa_tecnica")
    
    assert result is not None
    assert "marca" in result
    assert "modelo" in result

def test_extract_with_invalid_path():
    """Test error cuando imagen no existe"""
    extractor = APIExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract("/invalid/path.jpg", "placa_tecnica")
```

---

## 🔀 Proceso de Pull Request

### Antes de Enviar

**Checklist:**
- [ ] Código sigue PEP 8
- [ ] Todos los tests pasan (`pytest`)
- [ ] Agregaste tests para nuevo código
- [ ] Documentación actualizada
- [ ] Commits siguen Conventional Commits
- [ ] Rama está actualizada con `main`

### Crear Pull Request

1. Ve a tu fork en GitHub
2. Click en "Compare & pull request"
3. Llena el template:

```markdown
## Descripción
[Describe qué hace este PR]

## Tipo de Cambio
- [ ] Bug fix
- [ ] Nueva funcionalidad
- [ ] Breaking change
- [ ] Documentación

## Checklist
- [ ] Tests pasan localmente
- [ ] Código formateado con black
- [ ] Documentación actualizada

## Capturas de Pantalla (si aplica)
[Agregar screenshots de cambios en UI]

## Relacionado
Closes #123  <!-- Issue que cierra este PR -->
```

4. Click "Create pull request"

### Después de Enviar

- **Responde comentarios** rápidamente
- **Haz cambios solicitados** en nuevos commits
- **No forces push** después de review (preserva historial)
- **Se paciente**: Los maintainers revisarán cuando puedan

---

## 🐛 Reportar Bugs

### Antes de Reportar

1. **Busca** en [Issues existentes](https://github.com/tu-usuario/repo/issues)
2. **Lee** [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
3. **Verifica** que estás usando la última versión

### Template de Bug Report

```markdown
**Descripción del Bug**
[Descripción clara y concisa]

**Para Reproducir**
Pasos:
1. Ve a '...'
2. Click en '...'
3. Scroll down to '...'
4. Ver error

**Comportamiento Esperado**
[Qué esperabas que pasara]

**Capturas de Pantalla**
[Si aplica, agregar screenshots]

**Entorno:**
 - OS: [ej. Windows 11, Ubuntu 22.04]
 - Python: [ej. 3.10.5]
 - Versión: [ej. 1.0.0]

**Logs**
```
[Pegar logs relevantes aquí]
```

**Contexto Adicional**
[Cualquier otra información relevante]
```

---

## 💡 Sugerir Mejoras

### Feature Requests

```markdown
**¿Tu feature request está relacionado con un problema?**
[ej. "Me frustra que no pueda exportar a XML"]

**Describe la solución que te gustaría**
[Descripción clara de lo que quieres que pase]

**Alternativas consideradas**
[Otras soluciones que consideraste]

**Contexto Adicional**
[Screenshots, mockups, ejemplos de otras apps]
```

---

## 📞 Contacto

**¿Dudas sobre contribución?**

- 💬 **GitHub Discussions**: [Link](https://github.com/tu-usuario/repo/discussions)
- 📧 **Email**: tu-email@ejemplo.com
- 💬 **Discord**: [Unirse al servidor](https://discord.gg/...)

---

## 🎉 Reconocimiento

Todos los contribuidores son agregados automáticamente a:
- [CONTRIBUTORS.md](CONTRIBUTORS.md)
- README.md (sección "Contributors")

---

**¡Gracias por contribuir!** 🚀
