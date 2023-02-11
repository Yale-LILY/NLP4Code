import yaml


FILE_PATH = "finetuning/training_configs/spider_t5_finetuning.yaml"


def main() -> int:
    with open(FILE_PATH, "r") as f:
        d = yaml.safe_load(stream=f)
        print(d)
    return 0


if __name__ == "__main__":
    exit(main())
