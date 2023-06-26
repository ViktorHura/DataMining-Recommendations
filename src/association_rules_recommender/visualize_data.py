import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math


def main():
    data = pickle.load(open("cache/data.pickle", "rb"))

    dataNDI = pickle.load(open("cache/dataNDI.pickle", "rb"))

    for score in dataNDI:
        xpoints = np.array([res['K'] for res in dataNDI[score]])
        yticks_set = set()
        line_names = [f'min_confidence={k}' for k in dataNDI[score][0].keys() if k != 'K']
        plt.figure(dpi=150, figsize=[11, 15])
        plt.style.use('seaborn-v0_8-bright')
        for confidence in [k for k in dataNDI[score][0].keys() if k != 'K']:
            ypoints = np.array([d[confidence] for d in dataNDI[score]])
            for p in ypoints:
                yticks_set.add(p)
            plt.plot(xpoints, ypoints, linewidth=0.5)

        plt.xlabel("top_K")
        plt.ylabel(score.capitalize(), rotation=0, labelpad=20)
        plt.title(f"NDI algorithm, {score.capitalize()}-score comparison with different confidences")

        legend_loc = "lower right"
        # if score == 'precision':
        #     legend_loc = "upper right"

        legend = plt.legend(line_names, loc=legend_loc)
        for l in legend.get_lines():
            l.set_linewidth(5)

        plt.xticks(xpoints)

        # do not put ticks that are too close together, keep only the larger tick
        yticks = [max(yticks_set)]
        for i in reversed(sorted(yticks_set)[:-1]):
            if yticks[-1] - i > 0.0012:
                yticks.append(i)

        plt.yticks(np.array(yticks))

        plt.grid(axis='y', linewidth=0.5, linestyle='--', )
        plt.show()

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