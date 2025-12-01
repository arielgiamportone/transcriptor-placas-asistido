"""
Excel Image Extractor Module
Extracts images from Excel files and maps them to BarCodes
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from openpyxl import load_workbook
from openpyxl_image_loader import SheetImageLoader
from PIL import Image
import io
from loguru import logger
from datetime import datetime


class ExcelImageExtractor:
    """
    Extracts images from Excel files (.xlsx, .xls) and creates mapping to BarCodes
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize extractor
        
        Args:
            output_dir: Directory to save extracted images
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.extraction_stats = {
            'total_rows': 0,
            'rows_with_images': 0,
            'total_images': 0,
            'errors': 0,
        }
    
    def extract_from_excel(
        self,
        excel_path: str,
        barcode_column: str = 'BarCode',
        image_start_column: str = 'BJ',
        sheet_name: Optional[str] = None,
        use_file_paths: bool = True
    ) -> pd.DataFrame:
        """
        Extract all images from Excel file and create mapping DataFrame
        
        Args:
            excel_path: Path to Excel file (.xlsx or .xls)
            barcode_column: Name of the column containing barcodes
            image_start_column: Column letter where images start (e.g., 'BJ')
            sheet_name: Name of sheet to process (None = first sheet)
            use_file_paths: If True, reads file paths from cells instead of embedded images
        
        Returns:
            DataFrame with columns: BarCode, image_paths (list), has_images (bool)
        """
        excel_path = Path(excel_path)
        
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
        logger.info(f"Starting image extraction from: {excel_path}")
        logger.info(f"BarCode column: {barcode_column}, Image start column: {image_start_column}")
        logger.info(f"Mode: {'File paths' if use_file_paths else 'Embedded images'}")
        
        # Determine file type
        if excel_path.suffix.lower() == '.xls':
            logger.info("Detected .xls file. Converting to .xlsx first...")
            excel_path = self._convert_xls_to_xlsx(excel_path)
        
        # Read Excel data
        df = pd.read_excel(excel_path, sheet_name=sheet_name or 0)
        self.extraction_stats['total_rows'] = len(df)
        
        logger.info(f"Loaded Excel with {len(df)} rows and {len(df.columns)} columns")
        
        # Validate barcode column exists
        if barcode_column not in df.columns:
            raise ValueError(f"BarCode column '{barcode_column}' not found in Excel. Available columns: {df.columns.tolist()}")
        
        # Check if we should use file paths mode
        if use_file_paths:
            return self._extract_from_file_paths(df, excel_path, barcode_column, image_start_column)
        
        # Original embedded images mode
        # Load workbook for image extraction
        wb = load_workbook(excel_path)
        sheet = wb[sheet_name] if sheet_name else wb.active
        
        # Try to load images
        try:
            image_loader = SheetImageLoader(sheet)
        except Exception as e:
            logger.warning(f"Could not initialize SheetImageLoader: {e}")
            logger.warning("Attempting alternative extraction method...")
            return self._extract_images_alternative(df, excel_path, barcode_column, sheet)
        
        # Extract images for each row
        results = []
        
        for idx, row in df.iterrows():
            excel_row_idx = idx + 2  # Excel rows are 1-indexed, +1 for header
            barcode = row[barcode_column]
            
            if pd.isna(barcode):
                logger.warning(f"Row {excel_row_idx}: No barcode found, skipping")
                results.append({
                    'BarCode': None,
                    'image_paths': [],
                    'has_images': False,
                    'error': 'missing_barcode'
                })
                continue
            
            # Extract images from this row
            image_paths = self._extract_row_images(
                image_loader=image_loader,
                row_idx=excel_row_idx,
                barcode=str(barcode),
                image_start_column=image_start_column,
                max_images=10  # Check up to 10 image columns
            )
            
            results.append({
                'BarCode': barcode,
                'image_paths': image_paths,
                'has_images': len(image_paths) > 0,
                'image_count': len(image_paths),
                'error': None
            })
            
            if len(image_paths) > 0:
                self.extraction_stats['rows_with_images'] += 1
                self.extraction_stats['total_images'] += len(image_paths)
        
        results_df = pd.DataFrame(results)
        
        # Merge with original DataFrame
        final_df = pd.merge(
            df,
            results_df[['BarCode', 'image_paths', 'has_images', 'image_count', 'error']],
            on='BarCode',
            how='left'
        )
        
        logger.info(f"Extraction complete:")
        logger.info(f"  Total rows: {self.extraction_stats['total_rows']}")
        logger.info(f"  Rows with images: {self.extraction_stats['rows_with_images']}")
        logger.info(f"  Total images extracted: {self.extraction_stats['total_images']}")
        logger.info(f"  Errors: {self.extraction_stats['errors']}")
        
        # Save metadata
        self._save_extraction_metadata(final_df, excel_path)
        
        return final_df
    
    def _extract_row_images(
        self,
        image_loader: SheetImageLoader,
        row_idx: int,
        barcode: str,
        image_start_column: str,
        max_images: int = 10
    ) -> List[str]:
        """
        Extract all images from a single row
        
        Args:
            image_loader: SheetImageLoader instance
            row_idx: Excel row index (1-indexed)
            barcode: BarCode for naming files
            image_start_column: Starting column letter
            max_images: Maximum number of image columns to check
        
        Returns:
            List of saved image file paths
        """
        image_paths = []
        start_col_idx = self._column_letter_to_index(image_start_column)
        
        for img_idx in range(max_images):
            col_idx = start_col_idx + img_idx
            col_letter = self._column_index_to_letter(col_idx)
            cell_ref = f'{col_letter}{row_idx}'
            
            try:
                if image_loader.image_in(cell_ref):
                    image = image_loader.get(cell_ref)
                    
                    # Save image
                    safe_barcode = self._sanitize_filename(barcode)
                    image_filename = f"{safe_barcode}_{img_idx+1}.jpg"
                    image_path = self.images_dir / image_filename
                    
                    # Convert to RGB if necessary (handle RGBA, P modes)
                    if image.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'P':
                            image = image.convert('RGBA')
                        background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                        image = background
                    elif image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Save with reasonable quality
                    image.save(str(image_path), 'JPEG', quality=95)
                    
                    image_paths.append(str(image_path))
                    logger.debug(f"Extracted image: {cell_ref} → {image_path}")
                
            except Exception as e:
                logger.warning(f"Error extracting image from {cell_ref}: {e}")
                self.extraction_stats['errors'] += 1
                continue
        
        if image_paths:
            logger.info(f"Row {row_idx} (BarCode: {barcode}): Extracted {len(image_paths)} image(s)")
        
        return image_paths
    
    def _extract_images_alternative(
        self,
        df: pd.DataFrame,
        excel_path: Path,
        barcode_column: str,
        sheet
    ) -> pd.DataFrame:
        """
        Alternative extraction method using direct sheet._images access
        
        Args:
            df: Original DataFrame
            excel_path: Path to Excel file
            barcode_column: BarCode column name
            sheet: openpyxl sheet object
        
        Returns:
            DataFrame with image mappings
        """
        logger.info("Using alternative extraction method (direct sheet._images)")
        
        results = []
        
        # Get all images from sheet
        if not hasattr(sheet, '_images') or not sheet._images:
            logger.warning("No images found in sheet")
            for idx, row in df.iterrows():
                results.append({
                    'BarCode': row[barcode_column],
                    'image_paths': [],
                    'has_images': False,
                    'image_count': 0,
                    'error': 'no_images_in_sheet'
                })
            return pd.DataFrame(results)
        
        # Group images by row
        images_by_row = {}
        for img in sheet._images:
            # Get anchor row
            row_idx = img.anchor._from.row + 1  # Convert to 1-indexed
            if row_idx not in images_by_row:
                images_by_row[row_idx] = []
            images_by_row[row_idx].append(img)
        
        logger.info(f"Found {len(sheet._images)} images across {len(images_by_row)} rows")
        
        # Process each row in DataFrame
        for idx, row in df.iterrows():
            excel_row_idx = idx + 2  # Excel 1-indexed + header
            barcode = row[barcode_column]
            
            if pd.isna(barcode):
                results.append({
                    'BarCode': None,
                    'image_paths': [],
                    'has_images': False,
                    'image_count': 0,
                    'error': 'missing_barcode'
                })
                continue
            
            # Get images for this row
            row_images = images_by_row.get(excel_row_idx, [])
            image_paths = []
            
            for img_idx, img in enumerate(row_images):
                try:
                    # Extract image data
                    image_data = img._data()
                    image = Image.open(io.BytesIO(image_data))
                    
                    # Save image
                    safe_barcode = self._sanitize_filename(str(barcode))
                    image_filename = f"{safe_barcode}_{img_idx+1}.jpg"
                    image_path = self.images_dir / image_filename
                    
                    # Convert to RGB
                    if image.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'P':
                            image = image.convert('RGBA')
                        background.paste(image, mask=image.split()[-1] if 'A' in image.mode else None)
                        image = background
                    elif image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image.save(str(image_path), 'JPEG', quality=95)
                    image_paths.append(str(image_path))
                    
                    self.extraction_stats['total_images'] += 1
                    
                except Exception as e:
                    logger.warning(f"Error extracting image {img_idx+1} from row {excel_row_idx}: {e}")
                    self.extraction_stats['errors'] += 1
            
            results.append({
                'BarCode': barcode,
                'image_paths': image_paths,
                'has_images': len(image_paths) > 0,
                'image_count': len(image_paths),
                'error': None
            })
            
            if len(image_paths) > 0:
                self.extraction_stats['rows_with_images'] += 1
        
        results_df = pd.DataFrame(results)
        final_df = pd.merge(df, results_df, on='BarCode', how='left')
        
        logger.info(f"Alternative extraction complete:")
        logger.info(f"  Total rows: {len(df)}")
        logger.info(f"  Rows with images: {self.extraction_stats['rows_with_images']}")
        logger.info(f"  Total images: {self.extraction_stats['total_images']}")
        
        return final_df
    
    def _convert_xls_to_xlsx(self, xls_path: Path) -> Path:
        """
        Convert .xls file to .xlsx
        
        Args:
            xls_path: Path to .xls file
        
        Returns:
            Path to converted .xlsx file
        """
        xlsx_path = xls_path.with_suffix('.xlsx')
        
        if xlsx_path.exists():
            logger.info(f"Using existing converted file: {xlsx_path}")
            return xlsx_path
        
        try:
            df = pd.read_excel(xls_path)
            df.to_excel(xlsx_path, index=False, engine='openpyxl')
            logger.info(f"Converted {xls_path.name} to {xlsx_path.name}")
            return xlsx_path
        except Exception as e:
            logger.error(f"Failed to convert .xls to .xlsx: {e}")
            raise
    
    @staticmethod
    def _column_letter_to_index(column_letter: str) -> int:
        """Convert column letter (A, B, ..., Z, AA, ...) to index (1, 2, ..., 26, 27, ...)"""
        index = 0
        for char in column_letter.upper():
            index = index * 26 + (ord(char) - ord('A') + 1)
        return index
    
    @staticmethod
    def _column_index_to_letter(column_index: int) -> str:
        """Convert column index (1, 2, ...) to letter (A, B, ...)"""
        letter = ''
        while column_index > 0:
            column_index -= 1
            letter = chr(column_index % 26 + ord('A')) + letter
            column_index //= 26
        return letter
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Remove invalid characters from filename"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename
    
    def _save_extraction_metadata(self, df: pd.DataFrame, excel_path: Path):
        """Save metadata about extraction"""
        metadata = {
            'source_file': str(excel_path),
            'extraction_timestamp': datetime.now().isoformat(),
            'statistics': self.extraction_stats,
            'output_directory': str(self.images_dir),
        }
        
        metadata_path = self.output_dir / 'extraction_metadata.json'
        
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadata saved to: {metadata_path}")
    
    def _extract_from_file_paths(
        self,
        df: pd.DataFrame,
        excel_path: Path,
        barcode_column: str,
        image_start_column: str
    ) -> pd.DataFrame:
        """
        Extract images by reading file paths from cells
        
        Args:
            df: DataFrame with Excel data
            excel_path: Path to Excel file (used to resolve relative paths)
            barcode_column: Name of barcode column
            image_start_column: Starting column letter or pattern (e.g., 'BJ' or 'Imagen')
        
        Returns:
            DataFrame with image mappings
        """
        excel_dir = excel_path.parent
        
        # Detect image columns: either by letter (BJ onwards) or by name pattern (Imagen 1, Imagen 2, etc.)
        image_columns = []
        
        # Method 1: Look for columns starting with "Imagen"
        imagen_cols = [col for col in df.columns if str(col).startswith('Imagen')]
        if imagen_cols:
            image_columns = imagen_cols
            logger.info(f"✅ Found {len(image_columns)} image columns by name pattern: {image_columns[:3]}...")
        else:
            # Method 2: Use column letter index (fallback)
            start_col_idx = self._column_letter_to_index(image_start_column)
            image_columns = list(df.columns[start_col_idx:])
            logger.info(f"Using columns from {image_start_column} ({start_col_idx}): {image_columns[:3]}...")
        
        results = []
        
        # Debug: Log first row to see what we're reading
        if len(df) > 0:
            first_row = df.iloc[0]
            logger.info(f"Debug - First row BarCode: {first_row[barcode_column]}")
            logger.info(f"Debug - Image columns found: {image_columns[:5]}")
            for col_name in image_columns[:3]:  # Show first 3 only
                cell_value = first_row[col_name]
                logger.info(f"Debug - Cell [{col_name}]: '{cell_value}' (type: {type(cell_value).__name__})")
        
        for idx, row in df.iterrows():
            barcode = row[barcode_column]
            image_paths = []
            
            # Get all image columns
            for col_name in image_columns:
                cell_value = row[col_name]
                
                # Check if cell contains a string that looks like a file path
                if pd.notna(cell_value) and isinstance(cell_value, str):
                    # Clean the path
                    file_path_str = str(cell_value).strip()
                    
                    # Skip if empty
                    if not file_path_str:
                        continue
                    
                    # Log first attempt to find file
                    if idx == 0:
                        logger.info(f"Debug - Trying to find: '{file_path_str}'")
                        logger.info(f"Debug - Excel dir: {excel_dir}")
                    
                    # Try to resolve the path
                    # First try relative to Excel file
                    potential_path = excel_dir / file_path_str
                    
                    if idx == 0:
                        logger.info(f"Debug - Checking: {potential_path} (exists: {potential_path.exists()})")
                    
                    # Try alternative locations if not found
                    if not potential_path.exists():
                        # Try absolute path
                        potential_path = Path(file_path_str)
                        if idx == 0:
                            logger.info(f"Debug - Checking absolute: {potential_path} (exists: {potential_path.exists()})")
                    
                    if not potential_path.exists():
                        # Try in workspace Data folder
                        workspace_data = Path(__file__).parent / "Data" / file_path_str
                        if workspace_data.exists():
                            potential_path = workspace_data
                            if idx == 0:
                                logger.info(f"Debug - Found in Data folder: {potential_path}")
                    
                    # Handle files with spaces in names (IMG_ 000127 vs IMG_000127)
                    if not potential_path.exists() and " " in file_path_str:
                        # Try removing space after IMG_
                        normalized = file_path_str.replace("IMG_ ", "IMG_")
                        potential_path = excel_dir / normalized
                        if idx == 0:
                            logger.info(f"Debug - Trying normalized (no space): {potential_path} (exists: {potential_path.exists()})")
                        
                        if not potential_path.exists():
                            # Try in Data folder too
                            potential_path = Path(__file__).parent / "Data" / normalized
                            if idx == 0:
                                logger.info(f"Debug - Trying Data folder normalized: {potential_path} (exists: {potential_path.exists()})")
                    
                    # Check if file exists
                    if potential_path.exists() and potential_path.is_file():
                        # Verify it's an image file
                        if potential_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                            image_paths.append(str(potential_path.absolute()))
                            logger.debug(f"Found image: {potential_path}")
                    else:
                        if idx == 0:
                            logger.warning(f"Image not found after all attempts: {file_path_str}")
            
            # Create result
            has_images = len(image_paths) > 0
            
            if has_images:
                self.extraction_stats['rows_with_images'] += 1
                self.extraction_stats['total_images'] += len(image_paths)
            
            results.append({
                'BarCode': barcode,
                'image_paths': image_paths,
                'image_count': len(image_paths),
                'has_images': has_images
            })
        
        final_df = pd.DataFrame(results)
        
        logger.info(f"Extraction complete:")
        logger.info(f"  Total rows: {self.extraction_stats['total_rows']}")
        logger.info(f"  Rows with images: {self.extraction_stats['rows_with_images']}")
        logger.info(f"  Total images extracted: {self.extraction_stats['total_images']}")
        logger.info(f"  Errors: {self.extraction_stats['errors']}")
        
        # Save metadata
        self._save_extraction_metadata(final_df, excel_path)
        
        return final_df
    
    def get_stats(self) -> Dict:
        """Get extraction statistics"""
        return self.extraction_stats.copy()


def main():
    """Test the extractor"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python excel_image_extractor.py <path_to_excel>")
        sys.exit(1)
    
    excel_path = sys.argv[1]
    output_dir = Path("output")
    
    extractor = ExcelImageExtractor(output_dir)
    df = extractor.extract_from_excel(excel_path)
    
    print(f"\nExtraction complete!")
    print(f"Total rows: {len(df)}")
    print(f"Rows with images: {df['has_images'].sum()}")
    print(f"Total images: {df['image_count'].sum()}")
    
    # Save mapping CSV
    csv_path = output_dir / 'image_mapping.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nMapping saved to: {csv_path}")


if __name__ == "__main__":
    main()
