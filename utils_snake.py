from IPython import display
import matplotlib.pyplot as plt

plt.ion()


def Plot(scores, average_scores, last_100_average, plot_file_name):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training performance")
    plt.xlabel("Number of episodes")
    plt.ylabel("Score")
    plt.plot(scores, "o")
    plt.plot(average_scores)
    plt.plot(last_100_average)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(average_scores) - 1, average_scores[-1], str(average_scores[-1]))
    plt.text(len(last_100_average) - 1, last_100_average[-1], str(last_100_average[-1]))

    plt.legend(["Score", "average", "last_100_average"], loc="upper left")

    plt.savefig(plot_file_name)

    plt.show(block=False)
    plt.pause(0.01)


def PrintEpisodeResult(
    score, average_score, last_100_average_score, num_of_episodes, record
):
    print(
        "Ep:",
        num_of_episodes,
        "Score:",
        score,
        "Record:",
        record,
        "Avg:",
        "%.5f" % (average_score),
        "Mov_avg_100:",
        "%.5f" % (last_100_average_score),
    )
