import argparse
import logging
from pathlib import Path
from export_model import export_kv_cache_model, quantize_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and quantize LLaMA model")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to save exported models"
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip model quantization step"
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Export model with KV cache
        logging.info("Exporting model...")
        model = export_kv_cache_model()
        model_path = args.output_dir / "llama_3_2_3b.mlpackage"
        model.save(str(model_path))
        logging.info(f"Model exported to {model_path}")
        
        if not args.skip_quantization:
            # Quantize model
            logging.info("Quantizing model...")
            quantized_model = quantize_model(model)
            quantized_path = args.output_dir / "llama_3_2_3b_int4.mlpackage"
            quantized_model.save(str(quantized_path))
            logging.info(f"Quantized model saved to {quantized_path}")
        
        logging.info("Export completed successfully!")
        
    except Exception as e:
        logging.error(f"Export failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
