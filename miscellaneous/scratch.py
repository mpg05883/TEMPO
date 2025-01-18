from omegaconf import OmegaConf


def main():
    config = OmegaConf.load("../configs/multiple_datasets.yml")
    print(type(config))


main()
