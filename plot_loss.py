#!/usr/bin/env python3
#Author: Ondrej Lukas - lukasond@fel.cvut.cz
import sys
import matplotlib
import json

if __name__ == '__main__':
	with open("run-MAE-tag-dev_AUC_PR.json", "r") as infile:
		mae = []
		timeline = []
		tmp = json.load(infile)
		for x in tmp:
			timeline.append(x[1])
			mae.append(x[2])
	with open("run-MSE-tag-dev_AUC_PR.json", "r") as infile:
		mse = []
		tmp = json.load(infile)
		for x in tmp:
			timeline.append(x[1])
			mse.append(x[2])

		plt.plot(timeline,mae)
		plt.show()