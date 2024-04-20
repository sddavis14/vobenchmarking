import matplotlib.pyplot as plt
import numpy as np


TRANSLATION: str = "trans"
ROTATION: str = "rot"


def plot_error(abs_err, err_1, err_2, err_3, err_4, err_type: str):
    order = np.arange(0, len(err_1), 1)
    if err_type != TRANSLATION:
        label1, label2, label3 = 'W Rotation Error', 'X Rotation Error', 'Y Rotation Error'
    else:
        label1, label2, label3 = 'X Translation Error', 'Y Translation Error', 'Z Translation Error'
    plt.plot(order, err_1, label=label1, color='blue')
    plt.plot(order, err_2, label=label2, color='green')
    plt.plot(order, err_3, label=label3, color='red')
    if err_type != TRANSLATION:
        plt.plot(order, err_4, label='Z Rotation Error', color='black')
    plt.xlabel("Time Sequence")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    if err_type != TRANSLATION:
        plt.title(f'Rotational Error: {abs_err}')
        plt.savefig('../plots/rot_error.png')
    else:
        plt.title(f'Translational Error: {abs_err}')
        plt.savefig('../plots/translation_error.png')
    plt.close()


def plot_trajectory(gt_1, gt_2, pred_1, pred_2, file: str, x_label: str, y_label: str, title: str):
    plt.plot(gt_1, gt_2, label='Ground Truth Pose', color='blue')
    plt.plot(pred_1, pred_2, label='Predicted Pose', color='red')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('../plots/'+file)
    plt.close()


def evaluate():
    predicted = np.genfromtxt('../results/predicted.csv', delimiter=',', invalid_raise=False)
    ground_truth = np.genfromtxt('../results/gt.csv', delimiter=',', invalid_raise=False)
    timestamps_pred = predicted[:, 7]
    timestamps_gt = ground_truth[:, 7]
    common_ts, common_pred_ts, common_gt_ts = np.intersect1d(timestamps_pred, timestamps_gt, return_indices=True)
    predicted = predicted[common_pred_ts]
    ground_truth = ground_truth[common_gt_ts]
    pred_x, pred_y, pred_z = predicted[:, 0], predicted[:, 1], predicted[:, 2]
    pred_qw, pred_qx, pred_qy, pred_qz = predicted[:, 3], predicted[:, 4], predicted[:, 5], predicted[:, 6]
    gt_x, gt_y, gt_z = ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2]
    gt_qw, gt_qx, gt_qy, gt_qz = ground_truth[:, 3], ground_truth[:, 4], ground_truth[:, 5], ground_truth[:, 6]

    plot_trajectory(gt_x, gt_y, pred_x, pred_y, 'xyplane.png', 'X-Direction', 'Y-Direction', 'X-Y Projection')
    plot_trajectory(gt_x, gt_z, pred_x, pred_z, 'xzplane.png', 'X-Direction', 'Z-Direction', 'X-Z Projection')
    plot_trajectory(gt_y, gt_z, pred_y, pred_z, 'yzplane.png', 'Y-Direction', 'Z-Direction', 'Y-Z Projection')

    t_x_err, t_y_err, t_z_err = (pred_x - gt_x) ** 2 / len(gt_x), \
                                (pred_y - gt_y) ** 2 / len(gt_y), (pred_z - gt_z) ** 2 / len(gt_z)
    r_w_err, r_x_err, r_y_err, r_z_err = (pred_qw - gt_qw) ** 2 / len(gt_qw), \
                                (pred_qx - gt_qx) ** 2 / len(gt_qx), \
        (pred_qy - gt_qy) ** 2 / len(gt_qy), (pred_qz - gt_qz) ** 2 / len(gt_qz)

    translation_err = t_x_err + t_y_err + t_z_err
    rotational_err = r_w_err + r_x_err + r_y_err + r_z_err

    plot_error(np.sum(translation_err), t_x_err, t_y_err, t_z_err, None, "trans")
    plot_error(np.sum(rotational_err), r_w_err, r_x_err, r_y_err, r_z_err, "rot")


if __name__ == '__main__':
    evaluate()