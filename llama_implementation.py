import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import coremltools as ct
import numpy as np

class BaselineLlamaForCausalLM(LlamaForCausalLM):
    """Baseline LlamaForCausalLM model without key/value caching."""
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            
        Returns:
            torch.Tensor: Model logits
            
        Raises:
            ValueError: If input shapes don't match
        """
        if input_ids.shape != attention_mask.shape:
            raise ValueError(f"Input shapes mismatch: {input_ids.shape} vs {attention_mask.shape}")
            
        out = super().forward(
            input_ids,
            attention_mask,
            use_cache=False,
        )
        return out.logits

def export_baseline_model(
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
    batch_size: int = 1,
    context_size: int = 2048,
) -> ct.models.MLModel:
    """Export the baseline model to Core ML format.
    
    Args:
        model_id: HuggingFace model ID
        batch_size: Batch size for input shape
        context_size: Context window size
        
    Returns:
        ct.models.MLModel: Exported Core ML model
        
    Raises:
        RuntimeError: If model conversion fails
    """
    try:
        torch_model = BaselineLlamaForCausalLM.from_pretrained(model_id).eval()
        
        # Set input shapes
        input_shape = (batch_size, context_size)
        
        # Trace PyTorch model
        example_inputs = (
            torch.zeros(input_shape, dtype=torch.int32),
            torch.zeros(input_shape, dtype=torch.int32),
        )
        
        traced_model = torch.jit.trace(
            torch_model,
            example_inputs=example_inputs,
        )
        
        # Convert to Core ML format
        inputs = [
            ct.TensorType(shape=input_shape, dtype=np.int32, name="inputIds"),
            ct.TensorType(shape=input_shape, dtype=np.int32, name="attentionMask"),
        ]
        
        outputs = [ct.TensorType(dtype=np.float16, name="logits")]
        
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=outputs,
            minimum_deployment_target=ct.target.macOS13,
            skip_model_load=True,
        )
        
        return mlmodel
    except Exception as e:
        raise RuntimeError(f"Model conversion failed: {str(e)}")
