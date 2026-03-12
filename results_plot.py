"""
Generates comparison plots from training results.
Auto-discovers the most recent run for each model variant in models/,
or accepts --output-dir to control where plots are saved.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = 'output_plots'
DEFAULT_DPI = 300

DEPTHS = (20, 32, 44, 56)
COLORS_PLAIN = ('darkorange', 'blue', 'red', 'green')
COLORS_RESIDUAL = ('purple', 'cyan', 'magenta', 'lime')


def discover_latest_run(model_name, models_dir='models'):
    """Find the most recent run directory for a given model variant."""
    model_dir = os.path.join(models_dir, model_name)
    if not os.path.isdir(model_dir):
        return None
    latest = None
    latest_mtime = 0
    for date_dir in os.listdir(model_dir):
        date_path = os.path.join(model_dir, date_dir)
        if not os.path.isdir(date_path):
            continue
        for time_dir in os.listdir(date_path):
            time_path = os.path.join(date_path, time_dir)
            if not os.path.isdir(time_path):
                continue
            errors_file = os.path.join(time_path, 'test_errors.npy')
            if os.path.isfile(errors_file):
                mtime = os.path.getmtime(errors_file)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest = time_path
    return latest


def save_plot(fig, path, output_dir=DEFAULT_OUTPUT_DIR, dpi=DEFAULT_DPI):
    """Save matplotlib figure to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, path)
    fig.savefig(full_path, dpi=dpi)
    print(f"Plot saved to {full_path}")


def plot_errors(ax, info_list, label_prefix=""):
    """
    Plot training and testing errors from .npy files.

    Args:
        ax: matplotlib axes to plot on.
        info_list: list of (path, label, color) tuples.
        label_prefix: prefix for legend labels.
    """
    ax.set_ylabel('Error (%)')
    ax.set_xlabel('Epoch')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for path, label_suffix, color in info_list:
        try:
            test_err_data = np.load(os.path.join(path, 'test_errors.npy'))
            train_err_data = np.load(os.path.join(path, 'train_errors.npy'))

            if test_err_data.shape[0] < 2 or train_err_data.shape[0] < 2:
                print(f"Warning: unexpected data format for '{label_suffix}'. Skipping.")
                continue

            if test_err_data.size == 0 or train_err_data.size == 0:
                print(f"Warning: empty data for '{label_suffix}'. Skipping.")
                continue

            ax.plot(test_err_data[0], test_err_data[1] * 100,
                    label=f'{label_prefix}{label_suffix} (Test)',
                    color=color, linestyle='-', linewidth=2)
            ax.plot(train_err_data[0], train_err_data[1] * 100,
                    label=f'{label_prefix}{label_suffix} (Train)',
                    color=color, linestyle='--', linewidth=2)

        except FileNotFoundError:
            print(f"Warning: missing data files in {path}. Skipping '{label_suffix}'.")
        except Exception as e:
            print(f"Error processing '{label_suffix}' in {path}: {e}. Skipping.")
    ax.legend(loc='best', fontsize='medium', frameon=True)


def plain_vs_residual(show=False, output_dir=DEFAULT_OUTPUT_DIR):
    """Plot plain-20 vs residual-20 (options A and B)."""
    info = []
    for suffix, name, color in [
        ('P-N', 'Plain-20', 'darkorange'),
        ('R-A', 'Residual-A-20', 'purple'),
        ('R-B', 'Residual-B-20', 'violet'),
    ]:
        path = discover_latest_run(f'CifarResNet-20-{suffix}')
        if path:
            info.append((path, name, color))

    if not info:
        print("No data found for plain vs residual comparison.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_errors(ax, info)
    ax.set_title('Plain vs. Residual Network Performance (20 layers)')
    fig.tight_layout()
    save_plot(fig, 'plain_vs_residual.png', output_dir=output_dir)
    if show:
        plt.show()
    plt.close(fig)


def side_by_side(show=False, output_dir=DEFAULT_OUTPUT_DIR):
    """Plot plain networks (left) vs residual networks (right) at multiple depths."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    plain_info = []
    for depth, color in zip(DEPTHS, COLORS_PLAIN):
        path = discover_latest_run(f'CifarResNet-{depth}-P-N')
        if path:
            plain_info.append((path, f'Plain-{depth}', color))
    plot_errors(axs[0], plain_info)
    axs[0].set_title('Plain Networks')

    res_info = []
    for depth, color in zip(DEPTHS, COLORS_RESIDUAL):
        path = discover_latest_run(f'CifarResNet-{depth}-R-A')
        if path:
            res_info.append((path, f'ResNet-{depth}', color))
    plot_errors(axs[1], res_info)
    axs[1].set_title('Residual Networks (Option A)')

    fig.suptitle('Plain vs. Residual Network Comparison by Depth', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, 'side_by_side.png', output_dir=output_dir)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training result plots.')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory for saved plots')
    args = parser.parse_args()

    plain_vs_residual(show=args.show, output_dir=args.output_dir)
    side_by_side(show=args.show, output_dir=args.output_dir)
