"""
Image Preprocessor Module
Advanced image preprocessing for industrial nameplates and SCADA codes
Handles rotation, reflection removal, contrast enhancement, etc.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from loguru import logger
from PIL import Image


class ImagePreprocessor:
    """
    Advanced image preprocessing for OCR optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or self._default_config()
        self.stats = {
            'processed': 0,
            'rotations_corrected': 0,
            'reflections_reduced': 0,
            'errors': 0,
        }
    
    def _default_config(self) -> Dict:
        """Default preprocessing configuration"""
        return {
            'enabled': True,
            'steps': [
                'rotation_correction',
                'perspective_correction',
                'reflection_reduction',
                'denoising',
                'contrast_enhancement',
                'sharpening',
                'adaptive_binarization'
            ],
            'rotation': {
                'method': 'minAreaRect',
                'angle_tolerance': 2.0
            },
            'reflection': {
                'enabled': True,
                'method': 'hsv_inpainting'
            },
            'contrast': {
                'method': 'clahe',
                'clip_limit': 2.0,
                'tile_grid_size': (8, 8)
            },
            'denoising': {
                'method': 'non_local_means',
                'h': 10
            },
            'save_preprocessed': True
        }
    
    def preprocess(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        return_both: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply full preprocessing pipeline to image
        
        Args:
            image_path: Path to input image
            output_path: Path to save preprocessed image (optional)
            return_both: Return both color and binary versions
        
        Returns:
            Tuple of (preprocessed_color, preprocessed_binary) images
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            original_img = img.copy()
            
            # Apply preprocessing steps
            steps = self.config.get('steps', [])
            
            for step in steps:
                if step == 'rotation_correction':
                    img = self.correct_rotation(img)
                elif step == 'perspective_correction':
                    img = self.correct_perspective(img)
                elif step == 'reflection_reduction':
                    if self.config.get('reflection', {}).get('enabled', True):
                        img = self.reduce_reflections(img)
                elif step == 'denoising':
                    img = self.denoise(img)
                elif step == 'contrast_enhancement':
                    img = self.enhance_contrast(img)
                elif step == 'sharpening':
                    img = self.sharpen(img)
            
            # Generate binary version for OCR
            binary = self.adaptive_binarization(img)
            
            # Save if requested
            if output_path and self.config.get('save_preprocessed', True):
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), img)
                
                # Also save binary version
                binary_path = output_path.with_stem(output_path.stem + '_binary')
                cv2.imwrite(str(binary_path), binary)
            
            self.stats['processed'] += 1
            
            if return_both:
                return img, binary
            else:
                return img, binary
            
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {e}")
            self.stats['errors'] += 1
            raise
    
    def correct_rotation(self, img: np.ndarray) -> np.ndarray:
        """
        Detect and correct image rotation using minAreaRect method
        
        Args:
            img: Input image
        
        Returns:
            Rotated image
        """
        method = self.config.get('rotation', {}).get('method', 'minAreaRect')
        angle_tolerance = self.config.get('rotation', {}).get('angle_tolerance', 2.0)
        
        if method == 'minAreaRect':
            angle = self._detect_rotation_minAreaRect(img)
        elif method == 'hough':
            angle = self._detect_rotation_hough(img)
        else:
            angle = 0
        
        # Only rotate if angle is significant
        if abs(angle) > angle_tolerance:
            img = self._rotate_image(img, angle)
            self.stats['rotations_corrected'] += 1
            logger.debug(f"Corrected rotation: {angle:.2f}°")
        
        return img
    
    def _detect_rotation_minAreaRect(self, img: np.ndarray) -> float:
        """
        Detect rotation angle using minimum area rectangle method
        
        Args:
            img: Input image
        
        Returns:
            Rotation angle in degrees
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Invert if needed (text should be dark)
        gray = cv2.bitwise_not(gray)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Find coordinates of all non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        if len(coords) < 5:  # Need at least 5 points
            return 0
        
        # Find minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        return angle
    
    def _detect_rotation_hough(self, img: np.ndarray) -> float:
        """
        Detect rotation using Hough line transform
        
        Args:
            img: Input image
        
        Returns:
            Rotation angle in degrees
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        
        if lines is None:
            return 0
        
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)
        
        # Return median angle
        return np.median(angles) if angles else 0
    
    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle
        
        Args:
            img: Input image
            angle: Rotation angle in degrees
        
        Returns:
            Rotated image
        """
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def correct_perspective(self, img: np.ndarray) -> np.ndarray:
        """
        Correct perspective distortion (for angled photos)
        
        Args:
            img: Input image
        
        Returns:
            Perspective-corrected image
        """
        # Simple implementation - can be enhanced with edge detection
        # For now, just return original (perspective correction is complex)
        return img
    
    def reduce_reflections(self, img: np.ndarray) -> np.ndarray:
        """
        Reduce specular reflections (critical for metallic nameplates)
        
        Args:
            img: Input image
        
        Returns:
            Image with reduced reflections
        """
        method = self.config.get('reflection', {}).get('method', 'hsv_inpainting')
        
        if method == 'hsv_inpainting':
            return self._reduce_reflections_hsv(img)
        else:
            return img
    
    def _reduce_reflections_hsv(self, img: np.ndarray) -> np.ndarray:
        """
        Reduce reflections using HSV analysis and inpainting
        
        Args:
            img: Input image (BGR)
        
        Returns:
            Image with reduced reflections
        """
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Detect bright areas with low saturation (typical reflections)
        highlight_mask = cv2.inRange(v, 200, 255) & cv2.inRange(s, 0, 50)
        
        # Dilate mask slightly
        kernel = np.ones((3, 3), np.uint8)
        highlight_mask = cv2.dilate(highlight_mask, kernel, iterations=1)
        
        # Inpaint highlighted regions
        result = cv2.inpaint(img, highlight_mask, 5, cv2.INPAINT_TELEA)
        
        # Blend with original (70% corrected, 30% original)
        result = cv2.addWeighted(result, 0.7, img, 0.3, 0)
        
        self.stats['reflections_reduced'] += 1
        
        return result
    
    def denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Apply denoising
        
        Args:
            img: Input image
        
        Returns:
            Denoised image
        """
        method = self.config.get('denoising', {}).get('method', 'non_local_means')
        h = self.config.get('denoising', {}).get('h', 10)
        
        if method == 'non_local_means':
            # Non-Local Means Denoising (best quality, slower)
            denoised = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
        elif method == 'gaussian':
            # Gaussian blur (faster, less quality)
            denoised = cv2.GaussianBlur(img, (5, 5), 0)
        elif method == 'bilateral':
            # Bilateral filter (good edge preservation)
            denoised = cv2.bilateralFilter(img, 9, 75, 75)
        else:
            denoised = img
        
        return denoised
    
    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE
        
        Args:
            img: Input image
        
        Returns:
            Contrast-enhanced image
        """
        method = self.config.get('contrast', {}).get('method', 'clahe')
        
        if method == 'clahe':
            clip_limit = self.config.get('contrast', {}).get('clip_limit', 2.0)
            tile_grid_size = tuple(self.config.get('contrast', {}).get('tile_grid_size', [8, 8]))
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        elif method == 'histogram_equalization':
            # Simple histogram equalization
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            enhanced = img
        
        return enhanced
    
    def sharpen(self, img: np.ndarray) -> np.ndarray:
        """
        Sharpen image using unsharp mask
        
        Args:
            img: Input image
        
        Returns:
            Sharpened image
        """
        # Gaussian blur
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        
        return sharpened
    
    def adaptive_binarization(self, img: np.ndarray) -> np.ndarray:
        """
        Apply adaptive binarization for OCR
        
        Args:
            img: Input image (color or grayscale)
        
        Returns:
            Binary image
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return binary
    
    def analyze_quality(self, img: np.ndarray) -> Dict[str, float]:
        """
        Analyze image quality metrics
        
        Args:
            img: Input image
        
        Returns:
            Dictionary with quality metrics
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Blur detection (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Contrast (standard deviation)
        contrast = gray.std()
        
        # Brightness (mean)
        brightness = gray.mean()
        
        # Text density estimation (ratio of edges)
        edges = cv2.Canny(gray, 50, 150)
        text_density = np.count_nonzero(edges) / edges.size
        
        return {
            'blur': laplacian_var,
            'contrast': contrast,
            'brightness': brightness,
            'text_density': text_density,
            'is_blurry': laplacian_var < 100,
            'is_low_contrast': contrast < 50,
            'complexity': 'simple' if laplacian_var > 100 and contrast > 50 else 'complex'
        }
    
    def get_stats(self) -> Dict:
        """Get preprocessing statistics"""
        return self.stats.copy()


def visualize_preprocessing(image_path: str, output_dir: str = 'debug_output'):
    """
    Visualize preprocessing steps for debugging
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save debug images
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = ImagePreprocessor()
    
    # Original
    img_original = cv2.imread(image_path)
    cv2.imwrite(str(output_dir / '1_original.jpg'), img_original)
    
    # Step by step
    img = img_original.copy()
    
    img = preprocessor.correct_rotation(img)
    cv2.imwrite(str(output_dir / '2_rotation_corrected.jpg'), img)
    
    img = preprocessor.reduce_reflections(img)
    cv2.imwrite(str(output_dir / '3_reflections_reduced.jpg'), img)
    
    img = preprocessor.denoise(img)
    cv2.imwrite(str(output_dir / '4_denoised.jpg'), img)
    
    img = preprocessor.enhance_contrast(img)
    cv2.imwrite(str(output_dir / '5_contrast_enhanced.jpg'), img)
    
    img = preprocessor.sharpen(img)
    cv2.imwrite(str(output_dir / '6_sharpened.jpg'), img)
    
    binary = preprocessor.adaptive_binarization(img)
    cv2.imwrite(str(output_dir / '7_binary.jpg'), binary)
    
    # Quality analysis
    quality = preprocessor.analyze_quality(img_original)
    print(f"\nQuality Analysis:")
    for key, value in quality.items():
        print(f"  {key}: {value}")
    
    print(f"\nDebug images saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_preprocessor.py <image_path> [output_dir]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'debug_output'
    
    visualize_preprocessing(image_path, output_dir)
