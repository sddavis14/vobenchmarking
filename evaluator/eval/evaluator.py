import matplotlib.pyplot as plt
import numpy as np


TRANSLATION: str = "trans"
ROTATION: str = "rot"


def quat_to_eulers(x, y, z, w):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x ** 2 + y ** 2)
    roll_x = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch_y = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y ** 2 + z ** 2)
    yaw_z = np.arctan2(siny_cosp, cosy_cosp)

    return np.unwrap(roll_x), np.unwrap(pitch_y), np.unwrap(yaw_z)


def plot_error(abs_err, err_1, err_2, err_3, timestamps, err_type: str):
    order = np.arange(0, len(err_1), 1)
    if err_type != TRANSLATION:
        label1, label2, label3 = 'Roll Error(rad)', 'Pitch Error(rad)', 'Yaw Error(rad)'
    else:
        label1, label2, label3 = 'X Translation Error(m)', 'Y Translation Error(m)', \
            'Z Translation Error(m)'
    plt.plot(order, err_1, label=label1, color='red')
    plt.plot(order, err_2, label=label2, color='green')
    plt.plot(order, err_3, label=label3, color='blue')
    plt.xlabel("Time Sequence")
    if err_type != TRANSLATION:
        plt.ylabel("Absolute Rotation Error(rad)")
    else:
        plt.ylabel("Absolute Translation Error(m)")
    plt.legend()
    plt.grid(True)
    if err_type != TRANSLATION:
        plt.title(f'Rotational Error (Euler Angles)')
        plt.tight_layout()
        plt.savefig('../plots/rot_error.png')
        print(f'Rotational Error = {abs_err} rad')
    else:
        plt.title(f'Translational Error (X, Y, Z)')
        plt.tight_layout()
        plt.savefig('../plots/translation_error.png')
        print(f'Translational Error = {abs_err} m')
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


def reduce(result_data):
    result_dict = dict()
    for result_tuple in result_data:
        result_dict[result_tuple[7]] = result_tuple
    return np.array(list(result_dict.values()))


def evaluate():
    predicted = np.genfromtxt('../results/predicted.csv', delimiter=',', invalid_raise=False)
    ground_truth = np.genfromtxt('../results/gt.csv', delimiter=',', invalid_raise=False)

    pred_time_aligned = reduce(predicted)
    gt_time_aligned = reduce(ground_truth)

    reduced_len = min(len(pred_time_aligned), len(gt_time_aligned))
    pred_time_aligned = pred_time_aligned[: reduced_len]
    gt_time_aligned = gt_time_aligned[: reduced_len]

    timestamps = ground_truth[:, 8]

    pred_x, pred_y, pred_z = predicted[:, 0], predicted[:, 1], predicted[:, 2]
    pred_qw, pred_qx, pred_qy, pred_qz = pred_time_aligned[:, 3], pred_time_aligned[:, 4],\
        pred_time_aligned[:, 5], pred_time_aligned[:, 6]
    pred_roll, pred_pitch, pred_yaw = quat_to_eulers(pred_qx, pred_qy, pred_qz, pred_qw)

    gt_x, gt_y, gt_z = ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2]
    gt_qw, gt_qx, gt_qy, gt_qz = gt_time_aligned[:, 3], gt_time_aligned[:, 4],\
        gt_time_aligned[:, 5], gt_time_aligned[:, 6]
    gt_roll, gt_pitch, gt_yaw = quat_to_eulers(gt_qx, gt_qy, gt_qz, gt_qw)

    plot_trajectory(gt_x, gt_y, pred_x, pred_y, 'xyplane.png', 'X-Direction (m)', 'Y-Direction (m)', 'X-Y Projection')
    plot_trajectory(gt_x, gt_z, pred_x, pred_z, 'xzplane.png', 'X-Direction (m)', 'Z-Direction (m)', 'X-Z Projection')
    plot_trajectory(gt_y, gt_z, pred_y, pred_z, 'yzplane.png', 'Y-Direction (m)', 'Z-Direction (m)', 'Y-Z Projection')

    t_x_err, t_y_err, t_z_err = abs(pred_time_aligned[:, 0] - gt_time_aligned[:, 0]), \
                                abs(pred_time_aligned[:, 1] - gt_time_aligned[:, 1]), \
        abs(pred_time_aligned[:, 2] - gt_time_aligned[:, 2])
    r_roll_err, r_pitch_err, r_yaw_err = abs(pred_roll - gt_roll), \
                                abs(pred_pitch - gt_pitch), \
        abs(pred_yaw - gt_yaw)

    translation_err = t_x_err + t_y_err + t_z_err
    rotational_err = r_roll_err + r_pitch_err + r_yaw_err

    plot_error(np.sum(translation_err)/len(translation_err), t_x_err, t_y_err, t_z_err, timestamps, "trans")
    plot_error(np.sum(rotational_err)/len(rotational_err), r_roll_err, r_pitch_err, r_yaw_err, timestamps, "rot")


if __name__ == '__main__':
    evaluate()