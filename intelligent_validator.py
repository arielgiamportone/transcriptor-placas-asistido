"""
Intelligent Validator - Validación cruzada y corrección inteligente de datos extraídos
Verifica coherencia de datos y completa información parcial usando contexto
"""

import re
from typing import Dict, Optional, List, Tuple
from loguru import logger


class IntelligentValidator:
    """
    Valida y corrige datos extraídos usando conocimiento del dominio industrial
    """
    
    # Base de conocimiento de marcas por país/región
    BRAND_DATABASE = {
        # Checoslovaquia / República Checa
        'checoslovaquia': ['MEZ', 'MOTOR', 'CKD', 'TOS'],
        'czechoslovakia': ['MEZ', 'MOTOR', 'CKD', 'TOS'],
        'czech': ['MEZ', 'MOTOR', 'CKD', 'TOS'],
        
        # Alemania
        'germany': ['SIEMENS', 'SEW', 'NORD', 'LENZE', 'VEM', 'BONFIGLIOLI'],
        'alemania': ['SIEMENS', 'SEW', 'NORD', 'LENZE', 'VEM', 'BONFIGLIOLI'],
        'deutschland': ['SIEMENS', 'SEW', 'NORD', 'LENZE', 'VEM', 'BONFIGLIOLI'],
        
        # Italia
        'italy': ['BONFIGLIOLI', 'MOTOVARIO', 'STM', 'CANTONI', 'MARELLI'],
        'italia': ['BONFIGLIOLI', 'MOTOVARIO', 'STM', 'CANTONI', 'MARELLI'],
        
        # Francia
        'france': ['LEROY SOMER', 'ALMO', 'MOTOREDUCTEUR'],
        'francia': ['LEROY SOMER', 'ALMO', 'MOTOREDUCTEUR'],
        
        # España
        'spain': ['SIEMENS', 'WEG', 'CANTONI'],
        'españa': ['SIEMENS', 'WEG', 'CANTONI'],
        
        # USA
        'usa': ['BALDOR', 'MARATHON', 'GE', 'WESTINGHOUSE', 'RELIANCE'],
        'united states': ['BALDOR', 'MARATHON', 'GE', 'WESTINGHOUSE', 'RELIANCE'],
        
        # China
        'china': ['WEG', 'Y2', 'Y3', 'YE2', 'YE3'],
        
        # Brasil
        'brazil': ['WEG', 'EBERLE', 'KOHLBACH'],
        'brasil': ['WEG', 'EBERLE', 'KOHLBACH'],
    }
    
    # Prefijos/sufijos conocidos de marcas
    BRAND_PATTERNS = {
        'MEZ': ['M.E.Z', 'MEZ', 'MOTOR MEZ'],
        'SIEMENS': ['SIEMENS', 'SIEMENS AG'],
        'WEG': ['WEG', 'W.E.G'],
        'SEW': ['SEW', 'SEW-EURODRIVE'],
    }
    
    def __init__(self):
        self.validation_log = []
    
    def validate_and_correct(
        self,
        extracted_data: Dict,
        texto_completo: Optional[str] = None,
        image_type: str = 'placa_tecnica'
    ) -> Tuple[Dict, List[str]]:
        """
        Valida y corrige datos extraídos usando CROSS-REFERENCIA entre campos legibles
        
        Args:
            extracted_data: Datos extraídos por OCR/API
            texto_completo: Texto completo detectado en la imagen
            image_type: Tipo de imagen procesada
        
        Returns:
            Tuple de (datos_corregidos, lista_de_correcciones)
        """
        corrected_data = extracted_data.copy()
        corrections = []
        
        if image_type != 'placa_tecnica':
            return corrected_data, corrections
        
        # 1. Validar coherencia de potencia (puede estar confundida con corriente)
        if corrected_data.get('potencia'):
            potencia_validada, pot_correction = self._validate_power(
                corrected_data.get('potencia'),
                texto_completo,
                corrected_data
            )
            if pot_correction:
                corrected_data['potencia'] = potencia_validada
                corrections.append(pot_correction)
        
        # 2. Inferir potencia desde MÚLTIPLES fuentes si falta
        if not corrected_data.get('potencia') and texto_completo:
            # Intenta desde datos eléctricos
            potencia_inferida, inference_msg = self._infer_power_from_electrical(
                texto_completo
            )
            if potencia_inferida:
                corrected_data['potencia'] = potencia_inferida
                corrected_data['potencia_inferida'] = True
                corrections.append(inference_msg)
            else:
                # Intenta desde RPM y frame (tamaño físico)
                potencia_rpm, rpm_msg = self._infer_power_from_rpm_frame(
                    corrected_data,
                    texto_completo
                )
                if potencia_rpm:
                    corrected_data['potencia'] = potencia_rpm
                    corrected_data['potencia_inferida_rpm'] = True
                    corrections.append(rpm_msg)
        
        # 3. Completar marca usando CUALQUIER contexto disponible
        if corrected_data.get('marca'):
            marca_completa, marca_correction = self._complete_brand(
                corrected_data.get('marca'),
                texto_completo,
                corrected_data  # Pasar otros campos para contexto adicional
            )
            if marca_correction:
                corrected_data['marca'] = marca_completa
                corrections.append(marca_correction)
        
        # 4. Inferir marca desde modelo/número de serie si falta
        if not corrected_data.get('marca') and texto_completo:
            marca_inferida, marca_msg = self._infer_brand_from_model(
                corrected_data,
                texto_completo
            )
            if marca_inferida:
                corrected_data['marca'] = marca_inferida
                corrected_data['marca_inferida'] = True
                corrections.append(marca_msg)
        
        # 5. Validar coherencia de datos eléctricos
        if texto_completo:
            electrical_corrections = self._validate_electrical_coherence(
                corrected_data,
                texto_completo
            )
            if electrical_corrections:
                corrections.extend(electrical_corrections)
        
        # 6. Inferir voltaje desde frecuencia y contexto regional
        if not corrected_data.get('voltaje') and texto_completo:
            voltaje_inferido, volt_msg = self._infer_voltage_from_context(
                corrected_data,
                texto_completo
            )
            if voltaje_inferido:
                corrected_data['voltaje'] = voltaje_inferido
                corrected_data['voltaje_inferido'] = True
                corrections.append(volt_msg)
        
        # Log correcciones
        if corrections:
            logger.info(f"Correcciones aplicadas: {len(corrections)}")
            for corr in corrections:
                logger.info(f"  - {corr}")
        
        return corrected_data, corrections
    
    def _validate_power(
        self,
        potencia: str,
        texto_completo: Optional[str],
        datos: Dict
    ) -> Tuple[str, Optional[str]]:
        """
        Valida que el valor de potencia sea coherente con otros datos eléctricos
        """
        if not texto_completo:
            return potencia, None
        
        # Detectar si la "potencia" es en realidad corriente (A, Amperes)
        if re.search(r'(\d+\.?\d*)\s*A(?:mp)?(?:eres)?(?:\s|$)', potencia, re.IGNORECASE):
            # Es corriente, no potencia
            corriente_match = re.search(r'(\d+\.?\d*)\s*A', potencia)
            if corriente_match:
                corriente = float(corriente_match.group(1))
                
                # Buscar voltaje en texto completo
                voltage_match = re.search(r'(\d{3})\s*[/-]\s*(\d{3})\s*V', texto_completo)
                if voltage_match:
                    v1 = int(voltage_match.group(1))
                    v2 = int(voltage_match.group(2))
                    
                    # Calcular potencia aproximada (P = V * I * sqrt(3) * factor)
                    # Para motor trifásico: P(kW) ≈ V * I * 1.732 * 0.8 / 1000
                    potencia_kw = (v1 * corriente * 1.732 * 0.8) / 1000
                    potencia_hp = potencia_kw * 1.34  # Conversión aproximada
                    
                    # Redondear a valores estándar de motores
                    potencias_standard = [0.25, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.5, 10, 15, 20]
                    potencia_hp_std = min(potencias_standard, key=lambda x: abs(x - potencia_hp))
                    
                    return f"{potencia_hp_std}HP", f"⚠️ Corregido: '{potencia}' era corriente, no potencia. Calculado: ~{potencia_hp_std}HP (basado en {v1}V, {corriente}A)"
        
        return potencia, None
    
    def _complete_brand(
        self,
        marca_parcial: str,
        texto_completo: Optional[str],
        otros_datos: Dict = None
    ) -> Tuple[str, Optional[str]]:
        """
        Completa marca parcial usando CUALQUIER contexto disponible:
        - Letras visibles + patrones de números de serie
        - Letras visibles + voltaje/frecuencia característicos
        - Letras visibles + frame size típico de cada fabricante
        """
        if not marca_parcial:
            return marca_parcial, None
        
        marca_parcial_upper = marca_parcial.upper().strip()
        texto_upper = texto_completo.upper() if texto_completo else ""
        otros_datos = otros_datos or {}
        
        # ESTRATEGIA 1: Buscar marca completa en el texto
        for brand_full, patterns in self.BRAND_PATTERNS.items():
            for pattern in patterns:
                if pattern.upper() in texto_upper:
                    if marca_parcial_upper in brand_full or brand_full.startswith(marca_parcial_upper):
                        return brand_full, f"✅ Marca completada: '{marca_parcial}' → '{brand_full}' (nombre completo encontrado en placa)"
        
        # ESTRATEGIA 2: Inferir desde características técnicas
        # Ejemplo: WEG usa frames 71-355, voltajes 220/380V
        modelo = otros_datos.get('modelo', '')
        frame = otros_datos.get('frame', '')
        voltaje = otros_datos.get('voltaje', texto_upper)
        
        # WEG: Frame W22, modelos con W, voltajes típicos Brasil
        if marca_parcial_upper in ['W', 'WE'] and ('W22' in texto_upper or 'W21' in modelo.upper()):
            return 'WEG', f"✅ Marca completada: '{marca_parcial}' → 'WEG' (identificado por serie W22/W21)"
        
        # SIEMENS: Frame 1LA, 1LE, voltajes europeos
        if marca_parcial_upper in ['S', 'SIE', 'SI'] and ('1LA' in texto_upper or '1LE' in texto_upper):
            return 'SIEMENS', f"✅ Marca completada: '{marca_parcial}' → 'SIEMENS' (identificado por serie 1LA/1LE)"
        
        # MEZ: Frames típicos checoslovacos, series 3A, 2A, voltajes 220/380V
        if marca_parcial_upper in ['M', 'ME', 'MEZ']:
            if re.search(r'[23]A\s*\d', texto_upper) or '3A' in texto_upper or '2A' in texto_upper:
                return 'MEZ', f"✅ Marca completada: '{marca_parcial}' → 'MEZ' (identificado por serie 3A/2A)"
            if re.search(r'3A\d{2,3}[A-Z]?', modelo.upper()):
                return 'MEZ', f"✅ Marca completada: '{marca_parcial}' → 'MEZ' (identificado por modelo 3A)"
        
        # BALDOR: Series L, M, EM típicas USA
        if marca_parcial_upper in ['B', 'BA', 'BAL'] and re.search(r'[LM]\d{3,4}[A-Z]', texto_upper):
            return 'BALDOR', f"✅ Marca completada: '{marca_parcial}' → 'BALDOR' (identificado por serie L/M/EM)"
        
        # SEW: R, S, F series, reductores
        if marca_parcial_upper in ['S', 'SE'] and ('R' in modelo.upper() or 'F' in modelo.upper() or 'REDUCTOR' in texto_upper):
            return 'SEW', f"✅ Marca completada: '{marca_parcial}' → 'SEW' (identificado por serie reductora)"
        
        # ESTRATEGIA 3: Buscar país/región como ÚLTIMO RECURSO
        for region, brands in self.BRAND_DATABASE.items():
            region_variations = [region.upper()]
            
            if 'CZECH' in region.upper():
                region_variations.extend(['CZECH', 'CZECHOSLOV', 'CESKOSLOVENSK', 'PRAHA', 'PRAGUE'])
            if 'GERMANY' in region.upper() or 'ALEMAN' in region.upper():
                region_variations.extend(['GERMAN', 'DEUTSCH', 'MADE IN GERMANY'])
            if 'ITALY' in region.upper() or 'ITALIA' in region.upper():
                region_variations.extend(['ITALY', 'ITALIA', 'MADE IN ITALY'])
            if 'BRAZIL' in region.upper() or 'BRASIL' in region.upper():
                region_variations.extend(['BRAZIL', 'BRASIL', 'MADE IN BRAZIL'])
            
            if any(var in texto_upper for var in region_variations):
                for brand in brands:
                    if marca_parcial_upper in brand or brand.startswith(marca_parcial_upper):
                        return brand, f"✅ Marca completada: '{marca_parcial}' → '{brand}' (identificado por país: {region})"
        
        return marca_parcial, None
    
    def _validate_electrical_coherence(
        self,
        datos: Dict,
        texto_completo: str
    ) -> List[str]:
        """
        Valida coherencia entre datos eléctricos (voltaje, corriente, potencia, RPM)
        """
        warnings = []
        
        # Extraer datos eléctricos del texto
        voltage_match = re.search(r'(\d{2,3})\s*[/-]\s*(\d{3})\s*V', texto_completo)
        current_match = re.search(r'(\d+\.?\d*)\s*[/-]\s*(\d+\.?\d*)\s*A', texto_completo)
        rpm_match = re.search(r'(\d{3,4})\s*(?:r\.?p\.?m|rpm|RPM|v\.?p\.?m)', texto_completo, re.IGNORECASE)
        
        if voltage_match and current_match and datos.get('potencia'):
            v1 = int(voltage_match.group(1))
            i1 = float(current_match.group(1))
            
            potencia_str = datos.get('potencia', '')
            potencia_match = re.search(r'(\d+\.?\d*)\s*(HP|KW|CV)', potencia_str, re.IGNORECASE)
            
            if potencia_match:
                pot_valor = float(potencia_match.group(1))
                pot_unidad = potencia_match.group(2).upper()
                
                # Convertir todo a KW para comparar
                if pot_unidad == 'HP':
                    pot_kw = pot_valor * 0.746
                elif pot_unidad == 'CV':
                    pot_kw = pot_valor * 0.735
                else:
                    pot_kw = pot_valor
                
                # Calcular potencia esperada
                pot_calculada = (v1 * i1 * 1.732 * 0.8) / 1000
                
                # Verificar coherencia (tolerancia 30%)
                if abs(pot_kw - pot_calculada) / pot_calculada > 0.3:
                    warnings.append(
                        f"⚠️ Potencia inconsistente: {datos['potencia']} vs datos eléctricos ({v1}V, {i1}A → ~{pot_calculada:.2f}KW)"
                    )
        
        # Validar RPM típicos de motores
        if rpm_match:
            rpm = int(rpm_match.group(1))
            rpms_standard = [750, 1000, 1500, 3000]
            if rpm not in rpms_standard and not any(abs(rpm - std) < 50 for std in rpms_standard):
                warnings.append(f"⚠️ RPM inusual detectado: {rpm} (valores típicos: 750, 1000, 1500, 3000)")
        
        return warnings
    
    def _infer_power_from_electrical(
        self,
        texto_completo: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Infiere potencia a partir de datos eléctricos si falta
        """
        # Patrones más flexibles para voltaje
        voltage_match = re.search(r'(\d{2,3})\s*[/-]\s*(\d{3})\s*V', texto_completo)
        if not voltage_match:
            # Intenta con voltaje simple
            voltage_match = re.search(r'(\d{3})\s*V(?:\s|$)', texto_completo)
        
        # Patrones para corriente
        current_match = re.search(r'(\d+\.?\d*)\s*[/-]\s*(\d+\.?\d*)\s*A', texto_completo)
        if not current_match:
            # Intenta con corriente simple
            current_match = re.search(r'(\d+\.?\d*)\s*A(?:\s|$|mp)', texto_completo)
        
        if voltage_match and current_match:
            # Obtener voltaje (usar el primero si hay dos)
            if len(voltage_match.groups()) >= 2:
                v1 = int(voltage_match.group(1))
            else:
                v1 = int(voltage_match.group(1))
            
            # Obtener corriente (usar la primera si hay dos)
            if len(current_match.groups()) >= 2:
                i1 = float(current_match.group(1))
            else:
                i1 = float(current_match.group(1))
            
            # Calcular potencia
            potencia_kw = (v1 * i1 * 1.732 * 0.8) / 1000
            potencia_hp = potencia_kw * 1.34
            
            # Redondear a valores estándar
            potencias_standard = [0.25, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.5, 10, 15, 20]
            potencia_hp_std = min(potencias_standard, key=lambda x: abs(x - potencia_hp))
            
            return f"~{potencia_hp_std}HP", f"💡 Potencia inferida: ~{potencia_hp_std}HP (calculado desde {v1}V, {i1}A)"
        
        return None, None
    
    def _infer_power_from_rpm_frame(self, datos: Dict, texto_completo: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Infiere potencia aproximada desde RPM + Frame Size
        Frames pequeños (56-90) = 0.5-3HP
        Frames medianos (100-132) = 3-10HP
        Frames grandes (160-225) = 15-50HP
        """
        rpm_match = re.search(r'(\d{3,4})\s*(?:r\.?p\.?m|rpm|RPM)', texto_completo, re.IGNORECASE)
        frame_match = re.search(r'(?:FRAME|CARCASA|TAMAÑO)\s*:?\s*(\d{2,3})', texto_completo, re.IGNORECASE)
        
        if not frame_match:
            frame_match = re.search(r'\b(56|63|71|80|90|100|112|132|160|180|200|225)\s*(?:M|L|S)?\b', texto_completo)
        
        if rpm_match and frame_match:
            rpm = int(rpm_match.group(1))
            frame = int(frame_match.group(1))
            
            # Mapeo frame→potencia aproximada
            if frame <= 80:
                potencia_est = 1.0
            elif frame <= 100:
                potencia_est = 3.0
            elif frame <= 132:
                potencia_est = 7.5
            elif frame <= 180:
                potencia_est = 15.0
            else:
                potencia_est = 25.0
            
            return f"~{potencia_est}HP", f"💡 Potencia estimada: ~{potencia_est}HP (desde frame {frame} + {rpm}RPM)"
        
        return None, None
    
    def _infer_brand_from_model(self, datos: Dict, texto_completo: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Infiere marca desde patrones de modelo/serie
        """
        texto_upper = texto_completo.upper()
        
        # WEG: W22, W21, series
        if re.search(r'W2[0-9]', texto_upper):
            return 'WEG', f"✅ Marca inferida: 'WEG' (serie W2x detectada)"
        
        # SIEMENS: 1LA, 1LE
        if re.search(r'1L[AE]\d', texto_upper):
            return 'SIEMENS', f"✅ Marca inferida: 'SIEMENS' (serie 1LA/1LE detectada)"
        
        # BALDOR: EM, L, M series
        if re.search(r'[LME]{1,2}\d{4}', texto_upper):
            return 'BALDOR', f"✅ Marca inferida: 'BALDOR' (serie EM/L/M detectada)"
        
        # SEW: R, S, F (reductores)
        if re.search(r'[RSF]\d{2,3}', texto_upper) and 'REDUCTOR' in texto_upper:
            return 'SEW', f"✅ Marca inferida: 'SEW' (serie reductora detectada)"
        
        return None, None
    
    def _infer_voltage_from_context(self, datos: Dict, texto_completo: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Infiere voltaje desde frecuencia y contexto
        60Hz → Típicamente 220/440V (América)
        50Hz → Típicamente 220/380V (Europa/Asia)
        """
        freq_match = re.search(r'(50|60)\s*HZ', texto_completo, re.IGNORECASE)
        
        if freq_match:
            freq = int(freq_match.group(1))
            if freq == 60:
                return '220/440V', f"💡 Voltaje inferido: 220/440V (desde frecuencia 60Hz - típico América)"
            elif freq == 50:
                return '220/380V', f"💡 Voltaje inferido: 220/380V (desde frecuencia 50Hz - típico Europa/Asia)"
        
        return None, None
    
    def create_validation_prompt(self, texto_completo: str) -> str:
        """
        Crea un prompt adicional para segunda pasada de validación
        """
        return f"""VALIDACIÓN CRUZADA - USA CAMPOS LEGIBLES:

Texto completo detectado:
{texto_completo}

USA INFERENCIA CRUZADA para completar campos faltantes:

1. POTENCIA faltante/errónea:
   - Calcula desde Voltaje × Corriente × √3 × 0.8
   - Estima desde Frame + RPM (frame grande = mayor potencia)
   - Detecta si pusieron Corriente (A) en lugar de Potencia

2. MARCA faltante/parcial:
   - Busca serie/modelo característico (W22→WEG, 1LA→SIEMENS, EM→BALDOR)
   - Busca frame típico del fabricante
   - Solo si nada funciona, usa país como último recurso

3. VOLTAJE faltante:
   - Infiere desde frecuencia (60Hz→220/440V, 50Hz→220/380V)
   - Busca en texto completo números seguidos de V

4. RPM faltante:
   - Busca números de 3-4 dígitos cerca de rpm/RPM/r.p.m
   - Valores típicos: 750, 1000, 1500, 3000

5. COHERENCIA:
   - Verifica P ≈ V × I × 1.732 × 0.8 / 1000
   - RPM estándar vs inusual

Devuelve JSON corregido usando DATOS LEGIBLES como fuente principal."""


def test_validator():
    """Test del validador"""
    validator = IntelligentValidator()
    
    # Test caso 1: Corriente en lugar de potencia
    datos1 = {
        'marca': 'MOTOR',
        'potencia': '50A',
        'modelo': 'XYZ'
    }
    texto1 = "MOTOR ELECTRICO 220/380V 2.5/1.5A 1500 RPM"
    
    corrected1, corrections1 = validator.validate_and_correct(datos1, texto1)
    print("Test 1 - Corriente como potencia:")
    print(f"  Original: {datos1}")
    print(f"  Corregido: {corrected1}")
    print(f"  Correcciones: {corrections1}\n")
    
    # Test caso 2: Marca parcial
    datos2 = {
        'marca': 'MF',
        'potencia': '2HP',
        'modelo': 'ABC'
    }
    texto2 = "MF MOTOR CZECHOSLOVAKIA PRAGUE 220/380V 1500RPM"
    
    corrected2, corrections2 = validator.validate_and_correct(datos2, texto2)
    print("Test 2 - Marca parcial:")
    print(f"  Original: {datos2}")
    print(f"  Corregido: {corrected2}")
    print(f"  Correcciones: {corrections2}\n")


if __name__ == "__main__":
    test_validator()
