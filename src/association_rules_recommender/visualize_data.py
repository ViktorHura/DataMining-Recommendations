import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math


def main():
    data = pickle.load(open("data.pickle", "rb"))

    for score in data:
        xpoints = np.array([res['K'] for res in data[score]])

        for method in ['avg', 'max', 'wmx']:
            yticks_set = set()
            metric_names = []
            plt.figure(dpi=150, figsize=[11, 15])
            plt.style.use('seaborn-v0_8-bright')
            for metric in data[score][0]:
                if metric == "K" or metric[-3:] != method:
                    continue
                metric_names.append(metric[:-4])
                ypoints = np.array([res[metric] for res in data[score]])
                for p in ypoints:
                    yticks_set.add(p)

                plt.plot(xpoints, ypoints, linewidth=0.5)

            plt.xlabel("top_K")
            plt.ylabel(score.capitalize(), rotation=0, labelpad=20)
            plt.title(f"{score.capitalize()}-score, metric comparison, method={method}")

            legend_loc = "lower right"
            if score == 'precision':
                legend_loc = "upper right"

            legend = plt.legend(metric_names, loc=legend_loc)
            for l in legend.get_lines():
                l.set_linewidth(5)

            plt.xticks(xpoints)

            # do not put ticks that are too close together, keep only the larger tick
            yticks = [max(yticks_set)]
            for i in reversed(sorted(yticks_set)[:-1]):
                if yticks[-1]-i > 0.0012:
                    yticks.append(i)

            plt.yticks(np.array(yticks))

            plt.grid(axis='y', linewidth=0.5, linestyle='--',)
            plt.show()


if __name__ == '__main__':
    main()