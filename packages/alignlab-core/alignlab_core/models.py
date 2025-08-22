"""
Model provider abstractions for different inference backends.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = False
    num_return_sequences: int = 1
    stop_sequences: Optional[List[str]] = None


class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    def __init__(self, model_id: str, **kwargs):
        """
        Initialize the model provider.
        
        Args:
            model_id: Model identifier
            **kwargs: Provider-specific configuration
        """
        self.model_id = model_id
        self.config = kwargs
        self._model = None
        self._tokenizer = None
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass


class HuggingFaceProvider(ModelProvider):
    """HuggingFace model provider."""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.device = kwargs.get("device", "auto")
        self.dtype = kwargs.get("dtype", "auto")
        self.load_in_8bit = kwargs.get("load_in_8bit", False)
        self.load_in_4bit = kwargs.get("load_in_4bit", False)
    
    def load_model(self):
        """Load the model and tokenizer from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading model {self.model_id} from HuggingFace")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": self.device,
            }
            
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            else:
                model_kwargs["torch_dtype"] = self.dtype
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            logger.info(f"Successfully loaded model {self.model_id}")
            
        except ImportError:
            raise ImportError("transformers library is required for HuggingFace provider")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the loaded model."""
        if self._model is None or self._tokenizer is None:
            self.load_model()
        
        # Create generation config
        config = GenerationConfig(**kwargs)
        
        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to model device
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
                num_return_sequences=config.num_return_sequences,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                stop_sequences=config.stop_sequences
            )
        
        # Decode output
        generated_text = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()


class OpenAIProvider(ModelProvider):
    """OpenAI API model provider."""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.api_key = kwargs.get("api_key")
        if not self.api_key:
            import os
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
    
    def load_model(self):
        """No model loading needed for OpenAI API."""
        pass
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            # Create generation config
            config = GenerationConfig(**kwargs)
            
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Call API
            response = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                n=config.num_return_sequences,
                stop=config.stop_sequences
            )
            
            return response.choices[0].message.content.strip()
            
        except ImportError:
            raise ImportError("openai library is required for OpenAI provider")
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class VertexProvider(ModelProvider):
    """Google Vertex AI model provider."""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.project_id = kwargs.get("project_id")
        self.location = kwargs.get("location", "us-central1")
        
        if not self.project_id:
            import os
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        
        if not self.project_id:
            raise ValueError("Google Cloud project ID is required")
    
    def load_model(self):
        """No model loading needed for Vertex AI."""
        pass
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Vertex AI."""
        try:
            from vertexai.language_models import TextGenerationModel
            
            # Create generation config
            config = GenerationConfig(**kwargs)
            
            # Initialize model
            model = TextGenerationModel.from_pretrained(self.model_id)
            
            # Generate
            response = model.predict(
                prompt,
                max_output_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k
            )
            
            return response.text.strip()
            
        except ImportError:
            raise ImportError("google-cloud-aiplatform library is required for Vertex provider")
        except Exception as e:
            logger.error(f"Vertex AI error: {e}")
            raise


class VLLMProvider(ModelProvider):
    """vLLM model provider."""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.host = kwargs.get("host", "localhost")
        self.port = kwargs.get("port", 8000)
        self.api_base = kwargs.get("api_base", f"http://{self.host}:{self.port}")
    
    def load_model(self):
        """No model loading needed for vLLM server."""
        pass
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using vLLM server."""
        try:
            import requests
            
            # Create generation config
            config = GenerationConfig(**kwargs)
            
            # Prepare request
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "n": config.num_return_sequences,
                "stop": config.stop_sequences
            }
            
            # Make request
            response = requests.post(
                f"{self.api_base}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"vLLM server error: {response.text}")
            
            result = response.json()
            return result["choices"][0]["text"].strip()
            
        except ImportError:
            raise ImportError("requests library is required for vLLM provider")
        except Exception as e:
            logger.error(f"vLLM error: {e}")
            raise


def create_provider(provider_type: str, model_id: str, **kwargs) -> ModelProvider:
    """
    Factory function to create model providers.
    
    Args:
        provider_type: Type of provider ("hf", "openai", "vertex", "vllm")
        model_id: Model identifier
        **kwargs: Provider-specific configuration
        
    Returns:
        ModelProvider instance
    """
    if provider_type == "hf":
        return HuggingFaceProvider(model_id, **kwargs)
    elif provider_type == "openai":
        return OpenAIProvider(model_id, **kwargs)
    elif provider_type == "vertex":
        return VertexProvider(model_id, **kwargs)
    elif provider_type == "vllm":
        return VLLMProvider(model_id, **kwargs)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")

