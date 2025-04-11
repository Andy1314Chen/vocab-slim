import os
import copy
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class VocabSlim:
    """Vocabulary Slimming Tool for reducing the vocabulary size of pretrained language models

    Args:
        model_name_or_path (str): Path or name of the original model
        save_path (str): Path to save the slimmed model
        dataset_config (dict, optional): Configuration for dataset used for training new tokenizer
        target_vocab_size (int, optional): Target vocabulary size
        new_tokenizer_name_or_path (str, optional): Path or name of pretrained new tokenizer
        device (str, optional): Device to use, "auto" for auto-detect
    """

    def __init__(self, model_name_or_path, save_path, dataset_config=None, target_vocab_size=None, new_tokenizer_name_or_path=None, device="auto") -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing VocabSlim...")

        self.device = self._setup_device(device)
        self.logger.info(f"Using device: {self.device}")

        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.dataset_config = dataset_config

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=self.device if self.device == "cuda" else None
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception as e:
            self.logger.error(f"Failed to load model or tokenizer: {e}")
            raise

        self.new_tokenizer = self._initialize_tokenizer(dataset_config,
                                                        target_vocab_size,
                                                        new_tokenizer_name_or_path)

    def _setup_device(self, device):
        """Setup and validate device configuration"""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _initialize_tokenizer(self, dataset_config, target_vocab_size, new_tokenizer_name_or_path):
        """Initialize new tokenizer based on provided parameters"""
        if dataset_config is not None:
            return self._train_new_tokenizer(dataset_config, target_vocab_size)
        if new_tokenizer_name_or_path is not None:
            return AutoTokenizer.from_pretrained(new_tokenizer_name_or_path, trust_remote_code=True)
        raise ValueError(
            "Either dataset_config or new_tokenizer_name_or_path must be provided")

    def _train_new_tokenizer(self, dataset_config, target_vocab_size):
        """Train new tokenizer with reduced vocabulary size"""

        if target_vocab_size is None:
            raise ValueError(
                "target_vocab_size must be provided if dataset_config is provided")

        if target_vocab_size >= len(self.tokenizer.vocab):
            raise ValueError(
                f"Target vocab size ({target_vocab_size}) must be smaller than original vocab size ({len(self.tokenizer.vocab)})")

        try:
            dataset = load_dataset(
                dataset_config["name"],
                name=dataset_config.get("config", None),
                split=dataset_config.get("split", "train")
            )
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

        self.logger.info(
            f"Training new tokenizer with vocab size: {target_vocab_size}")
        batch_size = 1000

        text_column = dataset_config.get("text_column", "text")
        return self.tokenizer.train_new_from_iterator(self._batch_iterator(dataset, batch_size, text_column),
                                                      vocab_size=target_vocab_size)

    def _batch_iterator(self, dataset, batch_size, text_column):
        """Iterator for batched dataset processing"""
        total_batches = len(dataset) // batch_size
        for i in tqdm(range(0, len(dataset), batch_size),
                      desc="Training tokenizer",
                      total=total_batches):
            yield dataset[i: i + batch_size][text_column]

    def _calc_params(self, model):
        """Calculate total number of parameters in the model

        Args:
            model: The PyTorch model

        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in model.parameters())

    def _update_embedding(self, model, new2old_token_id):
        """Update embedding weights and biases for the new model"""
        for token_id, old_token_id in new2old_token_id.items():
            if isinstance(old_token_id, list):
                old_token_ids = torch.tensor(
                    old_token_id, device=self.model.device)
                model.model.embed_tokens.weight.data[token_id] = self.model.model.embed_tokens.weight.data[old_token_ids].mean(
                    dim=0)
                model.lm_head.weight.data[token_id] = self.model.lm_head.weight.data[old_token_ids].mean(
                    dim=0)

                if self.model.lm_head.bias is not None:
                    model.lm_head.bias.data[token_id] = self.model.lm_head.bias.data[old_token_ids].mean(
                    )
            else:
                model.model.embed_tokens.weight.data[token_id] = self.model.model.embed_tokens.weight.data[old_token_id]
                model.lm_head.weight.data[token_id] = self.model.lm_head.weight.data[old_token_id]
                if self.model.lm_head.bias is not None:
                    model.lm_head.bias.data[token_id] = self.model.lm_head.bias.data[old_token_id]

        return model

    def prune(self):
        """Execute vocabulary slimming operation"""
        new2old_token_id = self._build_token_mapping()

        old_params = self._calc_params(self.model)
        self.logger.info(
            f"Total params of original model: {old_params/1e6:.2f}M")

        new_model = self._create_new_model(new2old_token_id)
        self._log_compression_stats(old_params, new_model)
        self._save_models(new_model)

    def _build_token_mapping(self):
        """Build mapping between new and old token IDs"""
        old_vocab = self.tokenizer.vocab
        new_vocab = self.new_tokenizer.vocab
        new2old_token_id = {}

        for token, token_id in tqdm(new_vocab.items(), desc="Building token mapping"):
            if token not in old_vocab:
                token_decoded = self.new_tokenizer.decode([token_id])
                new2old_token_id[token_id] = self.tokenizer(
                    token_decoded).input_ids
            else:
                new2old_token_id[token_id] = old_vocab[token]

        return new2old_token_id

    def _create_new_model(self, new2old_token_id):
        """Create and configure new model with reduced vocabulary"""
        vocab_size = len(self.new_tokenizer)
        new_model = copy.deepcopy(self.model)
        new_model.resize_token_embeddings(vocab_size)

        with torch.no_grad():
            new_model = self._update_embedding(new_model, new2old_token_id)

        self._update_model_config(new_model, vocab_size)
        return new_model

    def _update_model_config(self, model, vocab_size):
        """Update model configuration with new vocabulary size"""
        bos_token_id = self.new_tokenizer.bos_token_id or vocab_size
        eos_token_id = self.new_tokenizer.eos_token_id or vocab_size + 2
        pad_token_id = self.new_tokenizer.pad_token_id or vocab_size

        model.config.update({
            "vocab_size": vocab_size,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id
        })

        model.generation_config.pad_token_id = pad_token_id
        model.generation_config.bos_token_id = bos_token_id
        model.generation_config.eos_token_id = eos_token_id

    def _save_models(self, model):
        """Save model and tokenizer"""
        try:
            model.save_pretrained(self.save_path)
            self.new_tokenizer.save_pretrained(self.save_path)
            self.logger.info(f"Model and tokenizer saved to {self.save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model or tokenizer: {e}")
            raise

    def _log_compression_stats(self, old_params, new_model):
        """Log compression statistics"""
        new_params = self._calc_params(new_model)
        vocab_ratio = len(self.new_tokenizer) / len(self.tokenizer) * 100
        param_ratio = new_params / old_params * 100

        self.logger.info(f"New model parameters: {new_params/1e6:.2f}M")
        self.logger.info(
            f"Vocabulary reduced to: {vocab_ratio:.2f}% of original")
        self.logger.info(
            f"Model parameters reduced to: {param_ratio:.2f}% of original")

    def check(self, text):
        """Compare outputs between original and slimmed models

        Args:
            text (str): Text to test
        """

        try:
            old_output_text = self._generate_text(
                self.model,
                self.tokenizer,
                text
            )

            new_model = AutoModelForCausalLM.from_pretrained(
                self.save_path).to(self.device)
            new_output_text = self._generate_text(
                new_model,
                self.new_tokenizer,
                text
            )

            self.logger.info("Comparison results:")
            self.logger.info(f"Original output: {old_output_text}")
            self.logger.info(f"Slimmed output: {new_output_text}")

        except Exception as e:
            self.logger.error(f"Error during model comparison: {e}")
            raise

    def _generate_text(self, model, tokenizer, text):
        """Generate text using specified model and tokenizer"""

        input_ids = tokenizer(
            text, return_tensors="pt").input_ids.to(self.device)
        output_ids = model.generate(input_ids, max_new_tokens=20)
        return tokenizer.batch_decode(output_ids)
