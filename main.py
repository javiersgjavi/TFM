import argparse
from TaxiExperiment import Taxi
from MontezumaExperiment import Montezuma


def main(args):
    env_id = int(args.environment[0])
    job = int(args.job[0])

    if env_id == 0:
        experiment = Montezuma()
    elif env_id == 1:
        experiment = Taxi()

    if job == 0:
        experiment.unified_learning(steps=5*10**5)
    elif job == 1:
        experiment.test()


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

    parser.add_argument(
        "-j",
        "--job",
        nargs=1,
        default=['0'],
        choices=['0', '1'],
        help="job: 0=train, 1=test"
    )
    args = parser.parse_args()
    main(args)
