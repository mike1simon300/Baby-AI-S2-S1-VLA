import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from tensorboard.backend.event_processing import event_accumulator

def load_scalar_events(log_dir, tag, max_x=None):
    """
    Load scalar events for a given tag from a TensorBoard log directory.
    Optionally truncate the data to the first max_x events.
    """
    if not os.path.exists(log_dir):
        print(f"Error: Log directory '{log_dir}' does not exist.")
        return None, None

    ea = event_accumulator.EventAccumulator(log_dir)
    try:
        ea.Reload()
    except Exception as e:
        print(f"Failed to load events from '{log_dir}': {e}")
        return None, None

    available_tags = ea.Tags().get("scalars", [])
    if tag not in available_tags:
        print(f"Warning: Tag '{tag}' not found in {log_dir}.")
        return None, None
    events = ea.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    if max_x is not None and len(steps) > max_x:
        steps = steps[:max_x]
        values = values[:max_x]
    return steps, values

def main():
    parser = argparse.ArgumentParser(
        description="Plot scalar tags from two TensorBoard log directories using a YAML config file."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML config file containing plot settings")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Required settings
    log_dir1 = config["log_dir1"]
    tags = config["tags"]
    label1 = config["label1"]
    output = config["output"]

    # Optional settings for second log
    log_dir2 = config.get("log_dir2", None)
    label2 = config.get("label2", "Log 2")

    # Optional additional plotting options
    max_x = config.get("max_x", None)             # Maximum number of data points per tag
    exponent = config.get("exponent", None)         # Exponent for x-axis scientific formatting (e.g., 6)
    x_label = config.get("x_label", "Training Steps")
    y_label = config.get("y_label", None)           # If not provided, defaults to the tag name capitalized

    # Font and line style options
    axis_fontsize = config.get("axis_fontsize", 12)       # Font size for axis and tick labels
    legend_fontsize = config.get("legend_fontsize", 10)     # Font size for the legend
    # Line style options for the curves:
    line_style1 = config.get("line_style1", "-")
    line_style2 = config.get("line_style2", "--")
    # Grid line options:
    grid_line_style = config.get("grid_line_style", "--")
    grid_line_width = config.get("grid_line_width", 0.5)
    # Professional font family:
    font_family = config.get("font_family", "serif")
    plt.rcParams["font.family"] = font_family

    # Determine subplot layout (using one subplot per tag)
    num_tags = len(tags)
    if num_tags == 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(nrows=num_tags, figsize=(8, 6 * num_tags))
        if not hasattr(axes, "__iter__"):
            axes = [axes]

    # Process each tag and plot curves
    for i, tag in enumerate(tags):
        ax = axes[i]
        # Load data from first log directory
        steps1, values1 = load_scalar_events(log_dir1, tag, max_x)
        if steps1 is not None:
            ax.plot(steps1, values1, label=label1, linestyle=line_style1)
        # Load data from second log directory, if provided
        if log_dir2 is not None:
            steps2, values2 = load_scalar_events(log_dir2, tag, max_x)
            if steps2 is not None:
                ax.plot(steps2, values2, label=label2, linestyle=line_style2)

        # Set x and y labels with the provided font sizes
        ax.set_xlabel(x_label, fontsize=axis_fontsize)
        ax.set_ylabel(y_label if y_label is not None else tag.capitalize(), fontsize=axis_fontsize)

        # Configure tick parameters for major and minor ticks
        ax.tick_params(axis="both", which="major", labelsize=axis_fontsize * 0.8)
        ax.tick_params(axis="both", which="minor", labelsize=axis_fontsize * 0.6)
        ax.minorticks_on()

        # Turn on grid lines for both major and minor ticks
        ax.grid(True, which="both", linestyle=grid_line_style, linewidth=grid_line_width)

        # Use scientific notation for the x-axis tick labels so that an offset text appears (e.g., "×10⁶")
        if exponent is not None:
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            formatter = ax.xaxis.get_major_formatter()
            if hasattr(formatter, 'set_offset_position'):
                formatter.set_offset_position('top')

        # Place legend inside the plot at the upper right
        ax.legend(loc="upper right", fontsize=legend_fontsize)

    # Remove overall title completely (no suptitle)
    plt.tight_layout()
    plt.savefig(output)
    plt.show()

if __name__ == "__main__":
    main()
