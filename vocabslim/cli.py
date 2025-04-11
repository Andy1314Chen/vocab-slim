import argparse
from core import VocabSlim


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Vocabulary Slimming Tool")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained model")
    parser.add_argument("--vocab_size", type=int, required=True,
                        help="Vocabulary size for new tokenizer (in thousands)")
    args = parser.parse_args()

    comVoc = VocabSlim(args.model_path,
                       save_path=f"{args.model_path}-{args.vocab_size}K",
                       dataset_config={"name": "wikitext",
                                       "config": "wikitext-103-raw-v1",
                                       "split": "train",
                                       "text_column": "text"},
                       target_vocab_size=args.vocab_size * 1000)
    comVoc.prune()
    comVoc.check("Hello, world!")


if __name__ == "__main__":
    main()
