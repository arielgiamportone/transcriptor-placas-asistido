"""
Shared Results Manager
Manages cross-tab result synchronization between Procesamiento Rápido and Transcripción Asistida
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger


class SharedResultsManager:
    """
    Manages shared results across different processing tabs
    Provides unified format and cross-tab synchronization
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize shared results manager
        
        Args:
            output_dir: Base output directory for results
        """
        self.output_dir = Path(output_dir)
        self.shared_results_file = self.output_dir / "shared_results.json"
        self.shared_results_file.parent.mkdir(parents=True, exist_ok=True)
    
    def load_shared_results(self) -> Dict[str, dict]:
        """
        Load all shared results from disk
        
        Returns:
            Dict mapping BarCode to consolidated results
            {
                'ABC123': {
                    'barcode': 'ABC123',
                    'results_by_type': {
                        'PLACA1_1': {...},
                        'SCADA1_1': {...}
                    },
                    'source_tabs': ['procesamiento_rapido', 'transcripcion_asistida'],
                    'last_updated': '2024-11-17T10:30:00',
                    'metadata': {...}
                }
            }
        """
        if not self.shared_results_file.exists():
            logger.debug(f"No shared results file found at {self.shared_results_file}")
            return {}
        
        try:
            with open(self.shared_results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert list to dict by barcode for easy lookup
            if isinstance(data, list):
                return {item['barcode']: item for item in data}
            return data
            
        except Exception as e:
            logger.error(f"Failed to load shared results: {e}")
            return {}
    
    def save_shared_results(self, results: Dict[str, dict]):
        """
        Save shared results to disk
        
        Args:
            results: Dict mapping BarCode to results
        """
        try:
            # Convert dict to list for storage
            results_list = list(results.values())
            
            with open(self.shared_results_file, 'w', encoding='utf-8') as f:
                json.dump(results_list, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved {len(results_list)} shared results to {self.shared_results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save shared results: {e}")
    
    def merge_result(self, barcode: str, tipo_label: str, data: dict, source_tab: str) -> dict:
        """
        Merge new result into shared results
        
        Args:
            barcode: BarCode identifier
            tipo_label: Type label (PLACA1_1, SCADA1_1, etc.)
            data: Extracted data dict
            source_tab: Source tab ('procesamiento_rapido' or 'transcripcion_asistida')
        
        Returns:
            Updated consolidated result for this barcode
        """
        # Load existing results
        all_results = self.load_shared_results()
        
        # Get or create entry for this barcode
        if barcode not in all_results:
            all_results[barcode] = {
                'barcode': barcode,
                'results_by_type': {},
                'source_tabs': [],
                'last_updated': datetime.now().isoformat(),
                'metadata': {}
            }
        
        barcode_entry = all_results[barcode]
        
        # Update results_by_type
        barcode_entry['results_by_type'][tipo_label] = {
            'marca': data.get('marca'),
            'modelo': data.get('modelo'),
            'numero_serie': data.get('numero_serie'),
            'año': data.get('año'),
            'potencia': data.get('potencia'),
            'codigo_scada_principal': data.get('codigo_scada_principal'),
            'confidence': data.get('overall_confidence', data.get('confidence', 0.0)),
            'method': data.get('method', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Track source tab
        if source_tab not in barcode_entry['source_tabs']:
            barcode_entry['source_tabs'].append(source_tab)
        
        barcode_entry['last_updated'] = datetime.now().isoformat()
        
        # Save back to disk
        self.save_shared_results(all_results)
        
        logger.info(f"Merged result for {barcode} [{tipo_label}] from {source_tab}")
        
        return barcode_entry
    
    def get_barcode_results(self, barcode: str) -> Optional[dict]:
        """
        Get all results for a specific barcode
        
        Args:
            barcode: BarCode identifier
        
        Returns:
            Consolidated result dict or None if not found
        """
        all_results = self.load_shared_results()
        return all_results.get(barcode)
    
    def import_from_procesamiento_rapido(self, batch_results: List[dict]) -> int:
        """
        Import results from Procesamiento Rápido format
        
        Args:
            batch_results: List of batch results from Procesamiento Rápido
        
        Returns:
            Number of results imported
        """
        all_results = self.load_shared_results()
        imported_count = 0
        
        for result in batch_results:
            barcode = result.get('BarCode')
            if not barcode:
                continue
            
            # Create or update entry
            if barcode not in all_results:
                all_results[barcode] = {
                    'barcode': barcode,
                    'results_by_type': {},
                    'source_tabs': [],
                    'last_updated': datetime.now().isoformat(),
                    'metadata': {}
                }
            
            entry = all_results[barcode]
            
            # Merge results_by_type
            for tipo_label, tipo_data in result.get('results_by_type', {}).items():
                entry['results_by_type'][tipo_label] = tipo_data
            
            # Track source
            if 'procesamiento_rapido' not in entry['source_tabs']:
                entry['source_tabs'].append('procesamiento_rapido')
            
            entry['last_updated'] = datetime.now().isoformat()
            imported_count += 1
        
        self.save_shared_results(all_results)
        logger.info(f"Imported {imported_count} results from Procesamiento Rápido")
        
        return imported_count
    
    def import_from_transcripcion_asistida(self, assisted_results: List[dict]) -> int:
        """
        Import results from Transcripción Asistida format
        
        Args:
            assisted_results: List of results from Transcripción Asistida
        
        Returns:
            Number of results imported
        """
        all_results = self.load_shared_results()
        imported_count = 0
        
        for result in assisted_results:
            barcode = result.get('BarCode')
            if not barcode:
                continue
            
            # Create or update entry
            if barcode not in all_results:
                all_results[barcode] = {
                    'barcode': barcode,
                    'results_by_type': {},
                    'source_tabs': [],
                    'last_updated': datetime.now().isoformat(),
                    'metadata': {}
                }
            
            entry = all_results[barcode]
            
            # Extract tipo information (may vary by format)
            # Typically Transcripción Asistida saves multiple images per barcode
            for img_key, img_data in result.items():
                if isinstance(img_data, dict) and 'tipo' in img_data:
                    tipo_label = img_data.get('tipo', 'UNKNOWN')
                    entry['results_by_type'][tipo_label] = {
                        'marca': img_data.get('marca'),
                        'modelo': img_data.get('modelo'),
                        'numero_serie': img_data.get('numero_serie'),
                        'año': img_data.get('año'),
                        'potencia': img_data.get('potencia'),
                        'codigo_scada_principal': img_data.get('codigo_scada_principal'),
                        'confidence': img_data.get('confidence', 0.0),
                        'method': img_data.get('method', 'manual'),
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Track source
            if 'transcripcion_asistida' not in entry['source_tabs']:
                entry['source_tabs'].append('transcripcion_asistida')
            
            entry['last_updated'] = datetime.now().isoformat()
            imported_count += 1
        
        self.save_shared_results(all_results)
        logger.info(f"Imported {imported_count} results from Transcripción Asistida")
        
        return imported_count
    
    def export_for_procesamiento_rapido(self) -> List[dict]:
        """
        Export results in Procesamiento Rápido format
        
        Returns:
            List of batch results compatible with Procesamiento Rápido
        """
        all_results = self.load_shared_results()
        
        batch_results = []
        for barcode, entry in all_results.items():
            batch_results.append({
                'BarCode': barcode,
                'results_by_type': entry['results_by_type'],
                'timestamp': entry['last_updated'],
                'source_tabs': entry['source_tabs']
            })
        
        return batch_results
    
    def get_statistics(self) -> dict:
        """
        Get statistics about shared results
        
        Returns:
            Dict with statistics
        """
        all_results = self.load_shared_results()
        
        total_barcodes = len(all_results)
        total_images = sum(len(entry['results_by_type']) for entry in all_results.values())
        
        by_source = {
            'procesamiento_rapido': 0,
            'transcripcion_asistida': 0,
            'both': 0
        }
        
        for entry in all_results.values():
            sources = entry['source_tabs']
            if 'procesamiento_rapido' in sources and 'transcripcion_asistida' in sources:
                by_source['both'] += 1
            elif 'procesamiento_rapido' in sources:
                by_source['procesamiento_rapido'] += 1
            elif 'transcripcion_asistida' in sources:
                by_source['transcripcion_asistida'] += 1
        
        return {
            'total_barcodes': total_barcodes,
            'total_images': total_images,
            'by_source': by_source,
            'file_path': str(self.shared_results_file)
        }
