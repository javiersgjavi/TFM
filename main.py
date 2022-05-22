import argparse
from TaxiExperiment import Taxi
from Montezuma import Montezuma


def main(args):
    env_id = int(args.environment[0])

    if env_id == 0:
        experiment = Montezuma()
    elif env_id == 1:
        experiment = Taxi()

    experiment.train(steps=7*10**5)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--environment",
        nargs=1,
        default=['0'],
        choices=['0', '1'],
        help="Environment to use"
    )
    args = parser.parse_args()
    main(args)
