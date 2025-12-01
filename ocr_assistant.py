"""
OCR Assistant Module
Provides OCR extraction with intelligent parsing and confidence scoring
Supports assisted transcription workflow
"""

import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from loguru import logger
import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not installed. OCR features will not work.")


class OCRAssistant:
    """
    OCR-based data extraction with confidence scoring
    Designed for human validation workflow
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize OCR Assistant
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR required. Install with: pip install easyocr")
        
        # Initialize EasyOCR
        languages = self.config.get('languages', ['es', 'en', 'de', 'fr', 'it'])
        use_gpu = self.config.get('gpu', False)
        
        logger.info(f"Initializing EasyOCR with languages: {languages}, GPU: {use_gpu}")
        self.reader = easyocr.Reader(languages, gpu=use_gpu)
        
        # Confidence thresholds
        self.high_confidence = self.config.get('high_confidence_threshold', 0.85)
        self.medium_confidence = self.config.get('medium_confidence_threshold', 0.60)
        
        # Pattern compilation cache
        self._compile_patterns()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'avg_processing_time': 0.0,
            'avg_confidence': 0.0,
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for field extraction"""
        self.patterns = {
            'serial_number': [
                re.compile(r'(?:S/?N|Serial|Serie|Número de Serie)[:\s]+([A-Z0-9\-]{5,20})', re.IGNORECASE),
                re.compile(r'\b([A-Z]{2}\d{8,12})\b'),
                re.compile(r'\b(N\s*\d{4,8})\b'),
                re.compile(r'\b(\d{2}-\d{5}-\d{6})\b'),
            ],
            'year': [
                re.compile(r'(?:Año|Year|Date|Fecha|Fabricación)[:\s]+(\d{4})', re.IGNORECASE),
                re.compile(r'\b(19\d{2}|20[0-2]\d)\b'),
            ],
            'brand': [
                re.compile(r'^([A-Z][a-zA-Z\s]{2,20})(?:\s|$)', re.MULTILINE),
            ],
            'model': [
                re.compile(r'(?:Model|Modelo|Mod)[:\s]+([A-Z0-9\-]{3,30})', re.IGNORECASE),
                re.compile(r'\b([A-Z0-9]{2,4}-[A-Z0-9\-]{3,20})\b'),
            ],
            'scada_code': [
                re.compile(r'\b([A-Z]{2,4}\d{6,10})\b'),
                re.compile(r'\b([A-Z0-9\-\.]{10,30})\b'),
            ],
        }
    
    def extract_from_image(
        self, 
        image_path: str,
        barcode: str,
        preprocessed_path: Optional[str] = None
    ) -> Dict:
        """
        Extract data from image using OCR
        
        Args:
            image_path: Path to original image
            barcode: BarCode identifier
            preprocessed_path: Optional path to preprocessed image (better OCR)
        
        Returns:
            Dictionary with extracted data and confidence scores
        """
        import time
        start_time = time.time()
        
        # Use preprocessed image if available (better OCR results)
        ocr_image_path = preprocessed_path if preprocessed_path else image_path
        
        logger.info(f"Processing OCR for {barcode}: {ocr_image_path}")
        
        try:
            # Run OCR
            results = self.reader.readtext(str(ocr_image_path))
            
            # Extract text blocks with confidence
            text_blocks = [(text, conf) for (bbox, text, conf) in results]
            full_text = ' '.join([text for text, _ in text_blocks])
            
            # Calculate average OCR confidence
            avg_ocr_conf = np.mean([conf for _, conf in text_blocks]) if text_blocks else 0.0
            
            # Classify image type
            image_type = self._classify_image_type(text_blocks)
            
            # Extract structured data
            extracted_data = {
                'barcode': barcode,
                'tipo_imagen': image_type,
                'full_text': full_text,
                'avg_ocr_confidence': float(avg_ocr_conf),
            }
            
            # Extract fields based on type
            if image_type == 'placa_tecnica':
                extracted_data.update(self._extract_nameplate_fields(text_blocks))
            elif image_type == 'codigo_scada':
                extracted_data.update(self._extract_scada_fields(text_blocks))
            else:
                # Unknown type, try extracting everything
                extracted_data.update(self._extract_nameplate_fields(text_blocks))
                extracted_data.update(self._extract_scada_fields(text_blocks))
            
            # Processing time
            processing_time = time.time() - start_time
            extracted_data['processing_time'] = processing_time
            
            # Update stats
            self._update_stats(processing_time, avg_ocr_conf)
            
            logger.info(f"OCR completed for {barcode} in {processing_time:.2f}s, confidence: {avg_ocr_conf:.2f}")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"OCR error for {barcode}: {e}")
            return {
                'barcode': barcode,
                'tipo_imagen': 'desconocido',
                'full_text': '',
                'error': str(e),
                'avg_ocr_confidence': 0.0,
            }
    
    def _classify_image_type(self, text_blocks: List[Tuple[str, float]]) -> str:
        """
        Classify image type based on OCR content
        
        Args:
            text_blocks: List of (text, confidence) tuples
        
        Returns:
            'placa_tecnica', 'codigo_scada', or 'desconocido'
        """
        full_text = ' '.join([text.lower() for text, _ in text_blocks])
        
        # Keywords for nameplate
        nameplate_keywords = ['modelo', 'model', 'serial', 'serie', 'marca', 'brand', 'año', 'year', 'fabricación']
        nameplate_score = sum(1 for kw in nameplate_keywords if kw in full_text)
        
        # Keywords for SCADA
        scada_keywords = ['fi', 'fic', 'tic', 'pic', 'li', 'code', 'tag']
        scada_score = sum(1 for kw in scada_keywords if kw in full_text)
        
        # Check for SCADA pattern (alphanumeric codes)
        scada_pattern_count = len(re.findall(r'\b[A-Z]{2,4}\d{5,10}\b', full_text.upper()))
        
        if scada_pattern_count >= 1 and scada_score >= nameplate_score:
            return 'codigo_scada'
        elif nameplate_score > 0:
            return 'placa_tecnica'
        else:
            return 'desconocido'
    
    def _extract_nameplate_fields(self, text_blocks: List[Tuple[str, float]]) -> Dict:
        """Extract nameplate fields (marca, modelo, SN, año)"""
        full_text = ' '.join([text for text, _ in text_blocks])
        
        fields = {}
        
        # Extract brand (marca)
        brand_result = self._extract_field('brand', full_text, text_blocks)
        fields['marca'] = brand_result['value']
        fields['marca_confidence'] = brand_result['confidence']
        fields['marca_source'] = brand_result.get('source', 'pattern')
        
        # Extract model (modelo)
        model_result = self._extract_field('model', full_text, text_blocks)
        fields['modelo'] = model_result['value']
        fields['modelo_confidence'] = model_result['confidence']
        fields['modelo_source'] = model_result.get('source', 'pattern')
        
        # Extract serial number
        sn_result = self._extract_field('serial_number', full_text, text_blocks)
        fields['numero_serie'] = sn_result['value']
        fields['numero_serie_confidence'] = sn_result['confidence']
        fields['numero_serie_source'] = sn_result.get('source', 'pattern')
        
        # Extract year
        year_result = self._extract_field('year', full_text, text_blocks)
        fields['año'] = year_result['value']
        fields['año_confidence'] = year_result['confidence']
        fields['año_source'] = year_result.get('source', 'pattern')
        
        return fields
    
    def _extract_scada_fields(self, text_blocks: List[Tuple[str, float]]) -> Dict:
        """Extract SCADA code (prefer longest)"""
        full_text = ' '.join([text for text, _ in text_blocks])
        
        fields = {}
        
        # Extract SCADA codes
        scada_codes = []
        for pattern in self.patterns['scada_code']:
            matches = pattern.findall(full_text)
            scada_codes.extend(matches)
        
        if scada_codes:
            # Select longest code (main SCADA code)
            longest_code = max(scada_codes, key=len)
            
            # Find confidence for this code
            code_confidence = self._find_text_confidence(longest_code, text_blocks)
            
            fields['codigo_scada_principal'] = longest_code
            fields['codigo_scada_principal_confidence'] = code_confidence
            fields['codigo_scada_principal_source'] = 'pattern'
        else:
            fields['codigo_scada_principal'] = None
            fields['codigo_scada_principal_confidence'] = 0.0
            fields['codigo_scada_principal_source'] = None
        
        return fields
    
    def _extract_field(
        self, 
        field_name: str, 
        full_text: str, 
        text_blocks: List[Tuple[str, float]]
    ) -> Dict:
        """
        Extract a specific field using patterns
        
        Returns:
            Dict with 'value', 'confidence', 'source'
        """
        patterns = self.patterns.get(field_name, [])
        
        for pattern in patterns:
            match = pattern.search(full_text)
            if match:
                value = match.group(1) if match.groups() else match.group(0)
                
                # Clean value
                value = value.strip()
                
                # Validate year
                if field_name == 'year':
                    try:
                        year = int(value)
                        if not (1950 <= year <= 2025):
                            continue
                    except ValueError:
                        continue
                
                # Find confidence for this text
                confidence = self._find_text_confidence(value, text_blocks)
                
                return {
                    'value': value,
                    'confidence': confidence,
                    'source': 'pattern',
                    'pattern': pattern.pattern
                }
        
        # No match found
        return {
            'value': None,
            'confidence': 0.0,
            'source': None
        }
    
    def _find_text_confidence(
        self, 
        search_text: str, 
        text_blocks: List[Tuple[str, float]]
    ) -> float:
        """
        Find OCR confidence for a specific text
        
        Args:
            search_text: Text to find
            text_blocks: List of (text, confidence) tuples
        
        Returns:
            Confidence score (0.0-1.0)
        """
        search_text_clean = search_text.lower().replace(' ', '')
        
        for text, conf in text_blocks:
            text_clean = text.lower().replace(' ', '')
            if search_text_clean in text_clean or text_clean in search_text_clean:
                return conf
        
        # Not found, return average confidence
        if text_blocks:
            return np.mean([conf for _, conf in text_blocks])
        else:
            return 0.0
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level (high, medium, low)
        
        Args:
            confidence: Confidence score (0.0-1.0)
        
        Returns:
            'high', 'medium', or 'low'
        """
        if confidence >= self.high_confidence:
            return 'high'
        elif confidence >= self.medium_confidence:
            return 'medium'
        else:
            return 'low'
    
    def get_confidence_color(self, confidence: float) -> str:
        """
        Get color code for confidence level
        
        Args:
            confidence: Confidence score (0.0-1.0)
        
        Returns:
            Color name for UI
        """
        level = self.get_confidence_level(confidence)
        colors = {
            'high': 'green',
            'medium': 'orange',
            'low': 'red'
        }
        return colors[level]
    
    def _update_stats(self, processing_time: float, confidence: float):
        """Update processing statistics"""
        n = self.stats['total_processed']
        
        # Running averages
        self.stats['avg_processing_time'] = (
            (self.stats['avg_processing_time'] * n + processing_time) / (n + 1)
        )
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * n + confidence) / (n + 1)
        )
        
        self.stats['total_processed'] += 1
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()


def test_ocr_assistant():
    """Test OCR Assistant"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_assistant.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    barcode = "TEST001"
    
    config = {
        'languages': ['es', 'en', 'de'],
        'gpu': False
    }
    
    assistant = OCRAssistant(config)
    
    print(f"Processing: {image_path}")
    print("-" * 60)
    
    result = assistant.extract_from_image(image_path, barcode)
    
    print("\n📊 Extracted Data:")
    print(f"Type: {result.get('tipo_imagen')}")
    print(f"OCR Confidence: {result.get('avg_ocr_confidence', 0):.2%}")
    print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
    print()
    
    if result.get('tipo_imagen') == 'placa_tecnica':
        print("Nameplate Fields:")
        fields = ['marca', 'modelo', 'numero_serie', 'año']
        for field in fields:
            value = result.get(field)
            conf = result.get(f'{field}_confidence', 0)
            level = assistant.get_confidence_level(conf)
            print(f"  {field}: {value} (confidence: {conf:.2%} - {level})")
    
    elif result.get('tipo_imagen') == 'codigo_scada':
        print("SCADA Code:")
        value = result.get('codigo_scada_principal')
        conf = result.get('codigo_scada_principal_confidence', 0)
        level = assistant.get_confidence_level(conf)
        print(f"  Code: {value} (confidence: {conf:.2%} - {level})")
    
    print(f"\n📝 Full Text:\n{result.get('full_text', '')[:200]}...")


if __name__ == "__main__":
    test_ocr_assistant()
