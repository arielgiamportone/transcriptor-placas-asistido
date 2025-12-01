"""
Configuration module for Industrial Image Extractor
Handles loading and validation of configuration from YAML and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from loguru import logger
import sys


class Config:
    """Central configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None, env_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config.yaml file
            env_path: Path to .env file
        """
        self.project_root = Path(__file__).parent.parent.parent
        
        # Load environment variables
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv(self.project_root / '.env')
        
        # Load YAML configuration
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = self.project_root / 'config.yaml'
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}. Using defaults.")
            self.config = self._default_config()
        else:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        
        # Override with environment variables where applicable
        self._apply_env_overrides()
        
        # Validate configuration
        self._validate()
        
        logger.info("Configuration loaded successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration if config file not found"""
        return {
            'processing': {'mode': 'hybrid', 'max_workers': 4},
            'output': {'base_dir': 'output'},
        }
    
    def _apply_env_overrides(self):
        """Override config values with environment variables"""
        # API keys
        if os.getenv('OPENAI_API_KEY'):
            self.config.setdefault('api', {})['api_key'] = os.getenv('OPENAI_API_KEY')
        
        if os.getenv('ANTHROPIC_API_KEY'):
            self.config.setdefault('api', {})['anthropic_key'] = os.getenv('ANTHROPIC_API_KEY')
        
        if os.getenv('GOOGLE_API_KEY'):
            self.config.setdefault('api', {})['google_key'] = os.getenv('GOOGLE_API_KEY')
        
        # Paths
        if os.getenv('EXCEL_INPUT_PATH'):
            self.config['excel_input_path'] = os.getenv('EXCEL_INPUT_PATH')
        
        if os.getenv('OUTPUT_BASE_DIR'):
            self.config['output']['base_dir'] = os.getenv('OUTPUT_BASE_DIR')
        
        # GPU
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('CUDA_VISIBLE_DEVICES')
        
        # Logging
        if os.getenv('LOG_LEVEL'):
            self.config['output']['reports']['log_level'] = os.getenv('LOG_LEVEL')
    
    def _validate(self):
        """Validate configuration values"""
        # Check processing mode
        valid_modes = ['api_only', 'local_only', 'hybrid']
        mode = self.config.get('processing', {}).get('mode', 'hybrid')
        if mode not in valid_modes:
            raise ValueError(f"Invalid processing mode: {mode}. Must be one of {valid_modes}")
        
        # Check API configuration if using API
        if mode in ['api_only', 'hybrid']:
            if not self.config.get('api', {}).get('api_key'):
                logger.warning("API mode selected but no API key found. Set OPENAI_API_KEY in .env")
        
        # Check GPU availability if using local mode
        if mode in ['local_only', 'hybrid']:
            if self.config.get('local', {}).get('ocr', {}).get('gpu', False):
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("GPU requested but CUDA not available. Will fall back to CPU.")
                        self.config['local']['ocr']['gpu'] = False
                        self.config['local']['vlm']['gpu'] = False
                except ImportError:
                    logger.warning("PyTorch not installed. Cannot use GPU.")
        
        # Validate thresholds
        thresholds = self.config.get('hitl', {}).get('thresholds', {})
        auto_accept = thresholds.get('auto_accept', 0.90)
        quick_review = thresholds.get('quick_review', 0.70)
        
        if auto_accept <= quick_review:
            raise ValueError("auto_accept threshold must be > quick_review threshold")
        
        logger.info(f"Configuration validated. Mode: {mode}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'api.model')
            default: Default value if key not found
        
        Returns:
            Configuration value
        
        Example:
            >>> config.get('api.model')
            'gpt-4o-mini-2024-07-18'
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        
        config[keys[-1]] = value
    
    @property
    def processing_mode(self) -> str:
        """Get processing mode"""
        return self.get('processing.mode', 'hybrid')
    
    @property
    def max_workers(self) -> int:
        """Get max workers for parallel processing"""
        return self.get('processing.max_workers', 4)
    
    @property
    def use_gpu(self) -> bool:
        """Check if GPU should be used"""
        return self.get('local.ocr.gpu', False)
    
    @property
    def output_dir(self) -> Path:
        """Get output directory path"""
        output_dir = Path(self.get('output.base_dir', 'output'))
        if not output_dir.is_absolute():
            output_dir = self.project_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        log_level = self.get('output.reports.log_level', 'INFO')
        
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level=log_level,
        )
        
        # Add file handler
        log_file = self.output_dir / 'reports' / 'processing.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level=log_level,
            rotation="100 MB",
            retention="30 days",
        )
        
        logger.info(f"Logging configured. Level: {log_level}, File: {log_file}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary"""
        return self.config.copy()
    
    def __repr__(self) -> str:
        return f"Config(mode={self.processing_mode}, output={self.output_dir})"


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None, env_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton pattern)
    
    Args:
        config_path: Path to config.yaml (only used on first call)
        env_path: Path to .env file (only used on first call)
    
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path, env_path)
        _config_instance.setup_logging()
    
    return _config_instance


def reset_config():
    """Reset configuration (mainly for testing)"""
    global _config_instance
    _config_instance = None


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print(f"Configuration loaded: {config}")
    print(f"Processing mode: {config.processing_mode}")
    print(f"Output directory: {config.output_dir}")
    print(f"Max workers: {config.max_workers}")
    print(f"Use GPU: {config.use_gpu}")
