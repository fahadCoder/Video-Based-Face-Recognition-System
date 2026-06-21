import matplotlib.pyplot as plt
import numpy as np

from cvproj_exc.classifier import NearestNeighborClassifier
from cvproj_exc.config import Config
from cvproj_exc.evaluation import OpenSetEvaluation


def main():
    false_alarm_rate_range = np.logspace(-3.0, 0, 1000, endpoint=False)

    train_data_file = Config.EVAL_TRAIN_DATA
    test_data_file = Config.EVAL_TEST_DATA

    classifier = NearestNeighborClassifier()

    evaluation = OpenSetEvaluation(
        classifier=classifier, false_alarm_rate_range=false_alarm_rate_range
    )
    evaluation.prepare_input_data(train_data_file, test_data_file)

    results = evaluation.run()

    far = results["false_alarm_rates"]
    dir_ = results["identification_rates"]
    tau = results["similarity_thresholds"]

    # -------------------------------------------------------
    # (A) FAR <= 1% and maximize identification rate
    mask_a = far <= 0.01
    idx_a_local = np.argmax(dir_[mask_a])           # index inside masked array
    idx_a = np.where(mask_a)[0][idx_a_local]        # convert to global index

    # (B) DIR >= 90% and minimize FAR
    mask_b = dir_ >= 0.90
    idx_b_local = np.argmin(far[mask_b])            # index inside masked array
    idx_b = np.where(mask_b)[0][idx_b_local]        # convert to global index

    print("A) FAR<=1% best DIR:", {"tau": float(tau[idx_a]), "FAR": float(far[idx_a]), "DIR": float(dir_[idx_a])})
    print("B) DIR>=90% lowest FAR:", {"tau": float(tau[idx_b]), "FAR": float(far[idx_b]), "DIR": float(dir_[idx_b])})
    # -------------------------------------------------------

    # Plot the DIR curve
    plt.semilogx(
        far,
        dir_,
        markeredgewidth=1,
        linewidth=3,
        linestyle="--",
        color="blue",
        label="DIR curve",
    )

    # Mark points A and B
    plt.scatter([far[idx_a]], [dir_[idx_a]], s=70, label="A: FAR<=1% best DIR")
    plt.scatter([far[idx_b]], [dir_[idx_b]], s=70, label="B: DIR>=90% min FAR")

    # Annotate points with values
    plt.annotate(
        f"A\nFAR={far[idx_a]:.4f}\nDIR={dir_[idx_a]:.3f}\nτ={tau[idx_a]:.3f}",
        xy=(far[idx_a], dir_[idx_a]),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->"),
    )
    plt.annotate(
        f"B\nFAR={far[idx_b]:.4f}\nDIR={dir_[idx_b]:.3f}\nτ={tau[idx_b]:.3f}",
        xy=(far[idx_b], dir_[idx_b]),
        xytext=(12, -45),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->"),
    )

    plt.grid(True, which="both")
    plt.axis([far[0], far[-1], 0, 1])
    plt.xlabel("False alarm rate")
    plt.ylabel("Identification rate")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()





# import matplotlib.pyplot as plt
# import numpy as np 

# from cvproj_exc.classifier import NearestNeighborClassifier
# from cvproj_exc.config import Config
# from cvproj_exc.evaluation import OpenSetEvaluation


# def main():
#     # The range of the false alarm rate in logarithmic space to draw DIR curves.
#     false_alarm_rate_range = np.logspace(-3.0, 0, 1000, endpoint=False)

#     # Pickle files containing embeddings and corresponding class labels for the
#     # training and the test dataset.
#     train_data_file = Config.EVAL_TRAIN_DATA
#     test_data_file = Config.EVAL_TEST_DATA

#     # We use a nearest neighbor classifier for this evaluation.
#     classifier = NearestNeighborClassifier()

#     # Prepare a new evaluation instance and feed training and test data into this evaluation.
#     evaluation = OpenSetEvaluation(
#         classifier=classifier, false_alarm_rate_range=false_alarm_rate_range
#     )
#     evaluation.prepare_input_data(train_data_file, test_data_file)

#     # Run the evaluation and retrieve the performance measures (identification rates and
#     # false alarm rates) on the test dataset.
#     results = evaluation.run()


#     # ******************************************************
#     far = results["false_alarm_rates"]
#     dir_ = results["identification_rates"]
#     tau = results["similarity_thresholds"]

#     # (A) FAR <= 1% and maximize identification rate
#     mask_a = far <= 0.01
#     idx_a = np.argmax(dir_[mask_a])
#     tau_a = tau[mask_a][idx_a]
#     far_a = far[mask_a][idx_a]
#     dir_a = dir_[mask_a][idx_a]
#     print("A) FAR<=1% best DIR:", {"tau": tau_a, "FAR": far_a, "DIR": dir_a})

#     # (B) DIR >= 90% and minimize FAR
#     mask_b = dir_ >= 0.90
#     idx_b = np.argmin(far[mask_b])
#     tau_b = tau[mask_b][idx_b]
#     far_b = far[mask_b][idx_b]
#     dir_b = dir_[mask_b][idx_b]
#     print("B) DIR>=90% lowest FAR:", {"tau": tau_b, "FAR": far_b, "DIR": dir_b})
#     # ******************************************************

#     # Plot the DIR curve.
#     plt.semilogx(
#         false_alarm_rate_range,
#         results["identification_rates"],
#         markeredgewidth=1,
#         linewidth=3,
#         linestyle="--",
#         color="blue",
#     )
#     plt.grid(True)
#     plt.axis(
#         [false_alarm_rate_range[0], false_alarm_rate_range[len(false_alarm_rate_range) - 1], 0, 1]
#     )
#     plt.xlabel("False alarm rate")
#     plt.ylabel("Identification rate")
#     plt.show()


# if __name__ == "__main__":
#     main()
