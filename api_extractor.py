"""
API Extractor
Extracts data using commercial API (OpenAI GPT-4o/Mini)
"""

import base64
import time
from typing import List, Optional
from pathlib import Path
from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not installed. API extraction will not work.")

from base_extractor import BaseExtractor, NameplateData

try:
    from intelligent_validator import IntelligentValidator
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
    logger.warning("IntelligentValidator not available. Validation will be skipped.")


class APIExtractor(BaseExtractor):
    """
    Extract data using OpenAI API with structured outputs
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize API extractor
        
        Args:
            config: Configuration dictionary with API settings
        """
        super().__init__(config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library required. Install with: pip install openai")
        
        # Get API key from config or environment
        api_key = self.config.get('api_key')
        if not api_key:
            import os
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Set in config or OPENAI_API_KEY env var")
        
        self.client = OpenAI(api_key=api_key)
        self.model = self.config.get('model', 'gpt-4o-mini-2024-07-18')
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout = self.config.get('timeout', 30)
        self.use_structured = self.config.get('structured_output', True)
        self.use_validation = self.config.get('intelligent_validation', True)
        
        # Initialize validator
        if VALIDATOR_AVAILABLE and self.use_validation:
            self.validator = IntelligentValidator()
            logger.info("IntelligentValidator enabled")
        else:
            self.validator = None
        
        logger.info(f"APIExtractor initialized with model: {self.model}")
    
    def extract(self, image_path: str, barcode: str, image_type: Optional[str] = None) -> NameplateData:
        """
        Extract data from single image using API
        
        Args:
            image_path: Path to image file
            barcode: BarCode identifier
            image_type: Type of image ('placa_tecnica' or 'codigo_scada'). If None, API will auto-detect
        
        Returns:
            NameplateData with extracted information
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return self._create_empty_result(str(image_path), barcode, "image_not_found")
        
        # Encode image to base64
        try:
            with open(image_path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return self._create_empty_result(str(image_path), barcode, f"encoding_error: {e}")
        
        # Create type-specific prompt
        if image_type == 'placa_tecnica':
            system_prompt = """Eres un experto en lectura de PLACAS TÉCNICAS industriales.

Tu tarea es extraer ÚNICAMENTE información de la placa técnica del equipo:
- **Marca**: Fabricante del equipo (ej: SCHNEIDER, ABB, SIEMENS)
- **Modelo**: Modelo o serie del equipo
- **Número de Serie (SN)**: Identificador único del equipo (letras+números, o con guiones)
- **Año de Fabricación**: Año de 4 dígitos entre 1950-2025
- **Potencia**: Potencia nominal con unidades (ej: 5.5KW, 200CV, 10KVA, 12000BTU, 3HP)

INSTRUCCIONES CRÍTICAS:
- Si la imagen está rotada, interprétala correctamente
- Si un campo no está visible, devuelve null para ese campo
- NO extraigas códigos SCADA, enfócate solo en datos de la placa técnica
- Busca texto cerca de etiquetas como: "Serial", "S/N", "Year", "Model", "Make"
- Para Año: debe ser un número entero de 4 dígitos

**VALIDACIÓN CRUZADA IMPORTANTE**:
- Si ves "A" (Amperes) NO lo pongas como potencia, deja potencia en null
- Si la marca está incompleta, busca contexto (país, región mencionada)
- Ejemplo: "MF" + "Checoslovaquia" → probablemente "MEZ"
- Verifica coherencia: si ves voltaje y corriente, la potencia debe ser razonable

Responde SOLO con JSON válido siguiendo el schema proporcionado."""

            user_prompt = f"""BarCode del activo: {barcode}

Analiza esta PLACA TÉCNICA y extrae:
1. Marca (fabricante)
2. Modelo
3. Número de Serie
4. Año de fabricación
5. Potencia con unidades (KW, CV, KVA, BTU, HP, etc.)

Ignora cualquier código SCADA o identificadores secundarios."""

        elif image_type == 'codigo_scada':
            system_prompt = """Eres un experto en lectura de CÓDIGOS SCADA industriales.

Tu tarea es encontrar el CÓDIGO SCADA PRINCIPAL de esta imagen:
- Busca el código MÁS LARGO visible
- Ignora códigos cortos entre paréntesis o en segunda línea (suelen ser obsoletos)
- Los códigos SCADA típicamente tienen formato: letras + números + puntos/guiones
- Suelen estar etiquetados como "TAG", "SCADA", "Control", "PLC"

INSTRUCCIONES CRÍTICAS:
- Si hay múltiples códigos, elige el MÁS LARGO (es el principal)
- Si un código está entre paréntesis y hay otro fuera, elige el de fuera
- Si la imagen está rotada, interprétala correctamente
- NO extraigas datos de placa técnica (marca, modelo, etc.)
- Si no ves ningún código SCADA claro, devuelve null

Responde SOLO con JSON válido siguiendo el schema proporcionado."""

            user_prompt = f"""BarCode del activo: {barcode}

Encuentra el CÓDIGO SCADA PRINCIPAL (el más largo) en esta imagen.
Ignora códigos secundarios o entre paréntesis.
Ignora datos de marca/modelo/serial."""

        else:
            # Generic prompt for auto-detection (extracts ALL available fields)
            system_prompt = """Eres un experto en lectura de documentación industrial.

Analiza esta imagen y extrae TODOS los datos disponibles:

**IMPORTANTE**: La imagen puede contener AMBOS tipos de información:
1. **PLACA TÉCNICA**: Marca, Modelo, Número de Serie, Año de fabricación, Potencia (con unidades: KW/CV/KVA/BTU/HP)
2. **CÓDIGO SCADA**: Código de referencia (busca el MÁS LARGO si hay varios)

INSTRUCCIONES CRÍTICAS:
- Extrae TODOS los campos que encuentres (placa técnica Y código SCADA)
- Si la imagen tiene ambos, llena ambos grupos de campos
- Si solo tiene uno, llena solo esos campos (marca el resto como null)
- Si la imagen está rotada, interprétala correctamente
- Para códigos SCADA: si hay múltiples, elige el MÁS LARGO (ignora códigos obsoletos entre paréntesis)
- Año debe ser un número de 4 dígitos entre 1950-2025
- Número de serie: formato típico es letras+números o números con guiones
- Potencia: debe incluir la unidad (ej: 5.5KW, 200CV, 10KVA, 12000BTU, 3HP)

**VALIDACIÓN CRUZADA IMPORTANTE**:
- NO confundas corriente (A, Amperes) con potencia
- Si ves "50A" NO lo pongas como potencia, déjalo null
- Si marca está incompleta ("MF", "SEW", etc) y ves país/región, infiere la marca completa
- Verifica coherencia: voltaje 220/380V con corriente 2.5A → potencia ~0.75HP aprox

**TIPO DE IMAGEN**:
- Si tiene solo placa técnica → 'placa_tecnica'
- Si tiene solo código SCADA → 'codigo_scada'
- Si tiene AMBOS → 'placa_tecnica' (preferencia para clasificación)

Responde SOLO con JSON válido siguiendo el schema proporcionado."""

            user_prompt = f"""BarCode del activo: {barcode}

Extrae TODOS los datos técnicos visibles:
- Si ves placa técnica: Marca, Modelo, Número de Serie, Año, Potencia
- Si ves código SCADA: Código SCADA principal (el más largo)
- Si ves AMBOS: extrae todos los campos

No omitas ningún dato visible."""
        
        # Log the prompt being sent
        logger.info(f"Sending API request for {barcode}:")
        logger.info(f"  Image type: {image_type or 'auto-detect'}")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  System prompt (first 150 chars): {system_prompt[:150]}...")
        logger.info(f"  User prompt: {user_prompt}")
        
        # Call API with retries
        for attempt in range(self.max_retries):
            try:
                if self.use_structured:
                    # Use structured outputs (guaranteed JSON conformity)
                    response = self._call_api_structured(image_b64, system_prompt, user_prompt)
                else:
                    # Use JSON mode (parse from text)
                    response = self._call_api_json_mode(image_b64, system_prompt, user_prompt)
                
                # Log the API response
                logger.info(f"API Response for {barcode} (attempt {attempt + 1}):")
                logger.info(f"  Tipo imagen: {response.get('tipo_imagen', 'N/A')}")
                logger.info(f"  Marca: {response.get('marca', 'null')}")
                logger.info(f"  Modelo: {response.get('modelo', 'null')}")
                logger.info(f"  Número Serie: {response.get('numero_serie', 'null')}")
                logger.info(f"  Año: {response.get('año', 'null')}")
                logger.info(f"  Potencia: {response.get('potencia', 'null')}")
                logger.info(f"  Código SCADA: {response.get('codigo_scada_principal', 'null')}")
                logger.info(f"  Confianza: {response.get('confidence', 0.0):.2f}")
                if response.get('texto_completo'):
                    logger.info(f"  Texto completo detectado: {response.get('texto_completo')[:200]}...")
                
                # Apply intelligent validation if enabled
                if self.validator and image_type != 'codigo_scada':
                    try:
                        validated_response, corrections = self.validator.validate_and_correct(
                            response,
                            response.get('texto_completo'),
                            response.get('tipo_imagen', 'placa_tecnica')
                        )
                        
                        if corrections:
                            logger.info(f"✅ Validación inteligente aplicada ({len(corrections)} correcciones):")
                            for correction in corrections:
                                logger.info(f"    {correction}")
                            
                            # Update response with validated data
                            response = validated_response
                            response['validation_applied'] = True
                            response['validation_corrections'] = corrections
                    except Exception as val_error:
                        logger.warning(f"Error en validación inteligente: {val_error}")
                
                # Parse response into NameplateData
                data = self._parse_api_response(response, barcode, str(image_path))
                
                self._update_stats(data, success=True)
                return data
                
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All API attempts failed for {image_path}")
                    result = self._create_empty_result(str(image_path), barcode, f"api_error: {e}")
                    self._update_stats(result, success=False)
                    return result
    
    def _call_api_structured(self, image_b64: str, system_prompt: str, user_prompt: str) -> dict:
        """
        Call API with structured outputs (OpenAI feature)
        
        Returns:
            Parsed dict from structured output
        """
        # Define response schema
        response_schema = {
            "type": "object",
            "properties": {
                "tipo_imagen": {
                    "type": "string",
                    "enum": ["placa_tecnica", "codigo_scada", "ambos", "desconocido"],
                    "description": "Tipo de imagen: 'placa_tecnica' si solo tiene placa, 'codigo_scada' si solo tiene SCADA, 'ambos' si tiene placa Y SCADA, 'desconocido' si no se puede determinar"
                },
                "marca": {"type": ["string", "null"], "description": "Marca del equipo"},
                "modelo": {"type": ["string", "null"], "description": "Modelo"},
                "numero_serie": {"type": ["string", "null"], "description": "Número de serie"},
                "año": {"type": ["integer", "null"], "minimum": 1950, "maximum": 2025, "description": "Año de fabricación"},
                "potencia": {"type": ["string", "null"], "description": "Potencia con unidades (ej: 5.5KW, 200CV, 10KVA, 12000BTU)"},
                "codigo_scada_principal": {"type": ["string", "null"], "description": "Código SCADA principal (el más largo)"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Confianza de la extracción"},
                "texto_completo": {"type": ["string", "null"], "description": "Todo el texto extraído de la imagen"}
            },
            "required": ["tipo_imagen", "marca", "modelo", "numero_serie", "año", "potencia", "codigo_scada_principal", "confidence", "texto_completo"],
            "additionalProperties": False
        }
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "high"  # Use high detail for better OCR
                    }}
                ]}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "nameplate_data",
                    "strict": True,
                    "schema": response_schema
                }
            },
            max_tokens=1000,
            timeout=self.timeout
        )
        
        # Parse JSON from response
        import json
        raw_content = response.choices[0].message.content
        
        # Log raw API response
        logger.debug(f"Raw API response (JSON): {raw_content}")
        
        return json.loads(raw_content)
    
    def _call_api_json_mode(self, image_b64: str, system_prompt: str, user_prompt: str) -> dict:
        """
        Call API with JSON mode (fallback)
        
        Returns:
            Parsed dict
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt + "\n\nResponde SOLO con JSON válido."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "high"
                    }}
                ]}
            ],
            response_format={"type": "json_object"},
            max_tokens=1000,
            timeout=self.timeout
        )
        
        import json
        content = response.choices[0].message.content
        
        # Log raw API response
        logger.debug(f"Raw API response (JSON mode): {content}")
        
        # Sometimes response has markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)
    
    def _parse_api_response(self, response: dict, barcode: str, image_path: str) -> NameplateData:
        """
        Parse API response into NameplateData
        
        Args:
            response: Dict from API
            barcode: BarCode
            image_path: Image path
        
        Returns:
            NameplateData object
        """
        return NameplateData(
            barcode=barcode,
            tipo_imagen=response.get('tipo_imagen', 'desconocido'),
            marca=response.get('marca'),
            modelo=response.get('modelo'),
            numero_serie=response.get('numero_serie'),
            año=response.get('año'),
            potencia=response.get('potencia'),
            codigo_scada_principal=response.get('codigo_scada_principal'),
            confidence=response.get('confidence', 0.8),
            metodo_extraccion="api",
            texto_completo=response.get('texto_completo'),
            raw_response=str(response),
            image_path=image_path
        )
    
    def extract_from_image_assisted(
        self, 
        image_path: str,
        barcode: str,
        preprocessed_path: Optional[str] = None,
        image_type: Optional[str] = None
    ) -> dict:
        """
        Extract data from image for assisted transcription workflow
        Returns data with confidence scores per field (compatible with OCR Assistant)
        
        Args:
            image_path: Path to original image
            barcode: BarCode identifier
            preprocessed_path: Optional path to preprocessed image (will use if provided)
            image_type: Type of image ('placa_tecnica' or 'codigo_scada'). If None, API will auto-detect
        
        Returns:
            Dict with extracted data and field-level confidence scores:
            {
                'barcode': str,
                'tipo_imagen': str,
                'marca': str | None,
                'modelo': str | None,
                'numero_serie': str | None,
                'año': int | None,
                'codigo_scada_principal': str | None,
                'overall_confidence': float,
                'field_confidence': {
                    'marca': float,
                    'modelo': float,
                    'numero_serie': float,
                    'año': float,
                    'codigo_scada_principal': float
                },
                'method': 'api',
                'model': str,
                'processing_time': float
            }
        """
        start_time = time.time()
        
        # Use preprocessed image if available (better quality)
        img_to_process = preprocessed_path if preprocessed_path and Path(preprocessed_path).exists() else image_path
        
        # Extract using standard API method with type-specific prompt
        result = self.extract(img_to_process, barcode, image_type=image_type)
        
        processing_time = time.time() - start_time
        
        # Convert to assisted format with field-level confidence
        # Since OpenAI gives overall confidence, we'll estimate per-field confidence
        overall_conf = result.confidence
        
        # Estimate field confidence based on whether field is populated and overall confidence
        field_confidence = {
            'marca': overall_conf if result.marca else 0.0,
            'modelo': overall_conf if result.modelo else 0.0,
            'numero_serie': overall_conf if result.numero_serie else 0.0,
            'año': overall_conf if result.año else 0.0,
            'potencia': overall_conf if result.potencia else 0.0,
            'codigo_scada_principal': overall_conf if result.codigo_scada_principal else 0.0,
        }
        
        return {
            'barcode': barcode,
            'tipo_imagen': result.tipo_imagen,
            'marca': result.marca,
            'modelo': result.modelo,
            'numero_serie': result.numero_serie,
            'año': result.año,
            'potencia': result.potencia,
            'codigo_scada_principal': result.codigo_scada_principal,
            'overall_confidence': overall_conf,
            'field_confidence': field_confidence,
            'method': 'api',
            'model': self.model,
            'processing_time': processing_time,
            'texto_completo': result.texto_completo,
            'raw_response': result.raw_response,  # Include raw API response for UI display
        }
    
    def extract_batch(self, image_paths: List[str], barcodes: List[str]) -> List[NameplateData]:
        """
        Extract data from multiple images
        
        Note: This implementation processes sequentially.
        For true batch API support, implement OpenAI Batch API separately.
        
        Args:
            image_paths: List of image paths
            barcodes: List of barcodes
        
        Returns:
            List of NameplateData
        """
        if len(image_paths) != len(barcodes):
            raise ValueError("Number of images must match number of barcodes")
        
        results = []
        
        for image_path, barcode in zip(image_paths, barcodes):
            result = self.extract(image_path, barcode)
            results.append(result)
            
            # Rate limiting (500 RPM default for tier 2)
            time.sleep(0.12)  # ~500 requests per minute
        
        return results
    
    def create_batch_file(self, image_paths: List[str], barcodes: List[str], output_path: str):
        """
        Create a batch file for OpenAI Batch API (50% discount)
        
        Args:
            image_paths: List of image paths
            barcodes: List of barcodes
            output_path: Path to save JSONL batch file
        """
        import jsonlines # type: ignore
        
        batch_requests = []
        
        for idx, (image_path, barcode) in enumerate(zip(image_paths, barcodes)):
            # Encode image
            with open(image_path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Create request
            request = {
                "custom_id": f"{barcode}_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "Extract data from industrial nameplate..."},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"BarCode: {barcode}"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high"
                            }}
                        ]}
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 1000
                }
            }
            
            batch_requests.append(request)
        
        # Save to JSONL
        with jsonlines.open(output_path, 'w') as writer:
            writer.write_all(batch_requests)
        
        logger.info(f"Batch file created with {len(batch_requests)} requests: {output_path}")
        logger.info("Upload with: openai.files.create() and submit with: openai.batches.create()")


def test_api_extractor():
    """Test the API extractor"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python api_extractor.py <image_path> <barcode>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    barcode = sys.argv[2]
    
    config = {
        'model': 'gpt-4o-mini-2024-07-18',
        'structured_output': True,
        'max_retries': 3
    }
    
    extractor = APIExtractor(config)
    
    print(f"Extracting data from: {image_path}")
    print(f"BarCode: {barcode}")
    print("-" * 60)
    
    result = extractor.extract(image_path, barcode)
    
    print("\nExtracted Data:")
    print(result.model_dump_json(indent=2))
    
    print(f"\nStats: {extractor.get_stats()}")


if __name__ == "__main__":
    test_api_extractor()
