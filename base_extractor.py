"""
Base Data Extractor
Abstract base class for all data extraction methods
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import json


class NameplateData(BaseModel):
    """
    Schema for extracted nameplate/SCADA data
    """
    barcode: str = Field(description="BarCode identifier from Excel")
    tipo_imagen: str = Field(description="Type: placa_tecnica, codigo_scada, or desconocido")
    marca: Optional[str] = Field(None, description="Brand/manufacturer")
    modelo: Optional[str] = Field(None, description="Model number")
    numero_serie: Optional[str] = Field(None, description="Serial number")
    año: Optional[int] = Field(None, ge=1950, le=2025, description="Manufacturing year")
    potencia: Optional[str] = Field(None, description="Power rating with units (e.g., 5.5KW, 200CV, 10KVA, 12000BTU)")
    codigo_scada_principal: Optional[str] = Field(None, description="Main SCADA code (longest)")
    
    # Metadata
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Extraction confidence")
    metodo_extraccion: str = Field(description="Extraction method used")
    texto_completo: Optional[str] = Field(None, description="Full extracted text")
    raw_response: Optional[str] = Field(None, description="Raw API/model response")
    
    # Processing info
    image_path: str = Field(description="Path to image processed")
    procesado_timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "barcode": "FI215002",
                "tipo_imagen": "codigo_scada",
                "marca": None,
                "modelo": None,
                "numero_serie": None,
                "año": None,
                "codigo_scada_principal": "FIL2150002",
                "confidence": 0.95,
                "metodo_extraccion": "api",
                "image_path": "/path/to/image.jpg",
            }
        }


class BaseExtractor(ABC):
    """
    Abstract base class for data extractors
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'avg_confidence': 0.0,
        }
    
    @abstractmethod
    def extract(self, image_path: str, barcode: str) -> NameplateData:
        """
        Extract data from image
        
        Args:
            image_path: Path to image file
            barcode: BarCode identifier
        
        Returns:
            NameplateData object with extracted information
        """
        pass
    
    @abstractmethod
    def extract_batch(self, image_paths: List[str], barcodes: List[str]) -> List[NameplateData]:
        """
        Extract data from multiple images
        
        Args:
            image_paths: List of image file paths
            barcodes: List of BarCode identifiers
        
        Returns:
            List of NameplateData objects
        """
        pass
    
    def _update_stats(self, data: NameplateData, success: bool):
        """Update extraction statistics"""
        self.stats['total_processed'] += 1
        if success:
            self.stats['successful'] += 1
            # Running average of confidence
            n = self.stats['successful']
            old_avg = self.stats['avg_confidence']
            self.stats['avg_confidence'] = (old_avg * (n - 1) + data.confidence) / n
        else:
            self.stats['failed'] += 1
    
    def get_stats(self) -> Dict:
        """Get extraction statistics"""
        return self.stats.copy()
    
    def _create_empty_result(self, image_path: str, barcode: str, error: str = None) -> NameplateData:
        """
        Create an empty result for failed extractions
        
        Args:
            image_path: Path to image
            barcode: BarCode
            error: Error message
        
        Returns:
            NameplateData with empty fields
        """
        return NameplateData(
            barcode=barcode,
            tipo_imagen="desconocido",
            confidence=0.0,
            metodo_extraccion=self.__class__.__name__,
            image_path=image_path,
            raw_response=error if error else "extraction_failed"
        )


def validate_extraction(data: NameplateData) -> bool:
    """
    Basic validation of extracted data
    
    Args:
        data: NameplateData to validate
    
    Returns:
        True if data seems valid, False otherwise
    """
    # Must have at least barcode and image path
    if not data.barcode or not data.image_path:
        return False
    
    # Must have identified type
    if data.tipo_imagen == "desconocido":
        return False
    
    # For placa_tecnica, should have at least 2 fields
    if data.tipo_imagen == "placa_tecnica":
        fields_present = sum([
            bool(data.marca),
            bool(data.modelo),
            bool(data.numero_serie),
            bool(data.año)
        ])
        return fields_present >= 2
    
    # For codigo_scada, must have the code
    if data.tipo_imagen == "codigo_scada":
        return bool(data.codigo_scada_principal)
    
    return True


def merge_extractions(extractions: List[NameplateData], method: str = "highest_confidence") -> NameplateData:
    """
    Merge multiple extractions of the same image (consensus voting)
    
    Args:
        extractions: List of NameplateData from different methods
        method: Merging strategy ("highest_confidence", "voting", "union")
    
    Returns:
        Single merged NameplateData
    """
    if len(extractions) == 0:
        raise ValueError("No extractions to merge")
    
    if len(extractions) == 1:
        return extractions[0]
    
    if method == "highest_confidence":
        # Return extraction with highest confidence
        return max(extractions, key=lambda x: x.confidence)
    
    elif method == "voting":
        # Majority vote on each field
        merged = extractions[0].model_copy()
        
        for field in ['marca', 'modelo', 'numero_serie', 'año', 'codigo_scada_principal']:
            values = [getattr(e, field) for e in extractions if getattr(e, field) is not None]
            if values:
                # Most common value
                merged.__setattr__(field, max(set(values), key=values.count))
        
        # Average confidence
        merged.confidence = sum(e.confidence for e in extractions) / len(extractions)
        merged.metodo_extraccion = "consensus_voting"
        
        return merged
    
    elif method == "union":
        # Union of all fields (prefer non-null values)
        merged = extractions[0].model_copy()
        
        for field in ['marca', 'modelo', 'numero_serie', 'año', 'codigo_scada_principal']:
            for extraction in extractions:
                value = getattr(extraction, field)
                if value is not None and getattr(merged, field) is None:
                    merged.__setattr__(field, value)
        
        # Max confidence
        merged.confidence = max(e.confidence for e in extractions)
        merged.metodo_extraccion = "union"
        
        return merged
    
    else:
        raise ValueError(f"Unknown merging method: {method}")


if __name__ == "__main__":
    # Test schema
    data = NameplateData(
        barcode="TEST001",
        tipo_imagen="placa_tecnica",
        marca="Air Jet",
        modelo="36-S-4-TRL-PL-PR",
        numero_serie="N 6612",
        año=2005,
        confidence=0.95,
        metodo_extraccion="test",
        image_path="/test/image.jpg"
    )
    
    print("Example NameplateData:")
    print(data.model_dump_json(indent=2))
    
    print(f"\nValidation: {validate_extraction(data)}")
