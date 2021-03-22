import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--path", type=str,
                    help="path of the dataset")

args = parser.parse_args()

cpt = sum([len(files) for r, d, files in os.walk(args.path)])
print(cpt)




