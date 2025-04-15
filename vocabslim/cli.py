import argparse
from core import VocabSlim


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Vocabulary Slimming Tool")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model")
    parser.add_argument("--vocab_size", type=int, required=True,
                        help="Vocabulary size for new tokenizer")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use, 'auto' for auto-detect")
    args = parser.parse_args()

    comVoc = VocabSlim(args.model_name_or_path,
                       save_path=f"{args.model_name_or_path}-{args.vocab_size // 1000}K",
                       dataset_config={"name": "wikitext",
                                       "config": "wikitext-103-raw-v1",
                                       "split": "train",
                                       "text_column": "text"},
                       target_vocab_size=args.vocab_size,
                       device=args.device)
    comVoc.prune()
    comVoc.check("What is the capital of France?")


if __name__ == "__main__":
    main()
