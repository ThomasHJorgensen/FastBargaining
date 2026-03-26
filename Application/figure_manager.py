from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib import transforms
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class FigureManager:
    def __init__(
        self,
        output_dir: str | Path = Path("figures/"),
        paper_size: str = "A4",
        file_ext: str = ".pdf",
        dpi: int = 300,
        use_latex: bool = True,
        cycle_linestyles: bool = False,
        cycle_markers: bool = False,
    ) -> None:
        """
        Initialize the FigureManager with output and style parameters.
        Args:
            output_dir (str | Path): Directory to save figures.
            paper_size (str): Paper size for the figure.
            file_ext (str): File extension for saved figures.
            dpi (int): Dots per inch for figure resolution.
            use_latex (bool): Whether to use LaTeX for text rendering.
            cycle_linestyles (bool): Whether to cycle through line styles.
            cycle_markers (bool): Whether to cycle through markers.
        """
        self.output_dir = Path(output_dir)
        self.paper_size = paper_size.lower()
        self.file_ext = file_ext
        self.dpi = dpi
        self.use_latex = use_latex
        self.cycle_linestyles = cycle_linestyles
        self.cycle_markers = cycle_markers

        # Enable LaTeX rendering for text
        if use_latex:
            plt.rc("text", usetex=True)
        else:
            plt.rc("text", usetex=False)

        # Set seaborn defaults
        sns.set_context("paper")  # Optimized for LaTeX documents
        sns.set_palette("deep")

        # Internal figure tracking
        self.fig = None
        self.axes = None
        self.n_rows = None
        self.n_cols = None
        self.n_subplots = None

    def _apply_custom_style(self) -> None:
        """
        The function `_apply_custom_style` sets custom style settings
        for matplotlib plots.
        """
        """Apply custom style settings."""
        # Axes properties
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 0.8
        plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.grid.axis"] = "y"
        plt.rcParams["axes.grid.which"] = "major"
        plt.rcParams["grid.linestyle"] = "dotted"
        plt.rcParams["grid.linewidth"] = 0.5
        plt.rcParams["grid.alpha"] = 1.0
        plt.rcParams["axes.facecolor"] = "white"

        # Font properties
        plt.rcParams["axes.titlesize"] = 11
        plt.rcParams["font.size"] = 10
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"

        # Lines properties
        plt.rcParams["lines.linewidth"] = 2
        plt.rcParams["lines.markersize"] = 6
        plt.rcParams["lines.markeredgewidth"] = 0.5

        # 12 colors for cycling
        colors = [
            "#EF476F",
            "#1082A8",
            "#FFD166",
            "#06EFB1",
            "#08485E",
            "#FF6F61",
            "#1B998B",
            "#C6C013",
            "#5A189A",
            "#A7C957",
            "#D4A5A5",
            "#3D348B",
        ]

        # Cycle through colors only
        # plt.rcParams['axes.prop_cycle'] = cycler('color', colors)
        # Cycle through colors and line styles and markers
        cycles = cycler("color", colors)
        if self.cycle_linestyles:
            cycles += cycler("linestyle", 3 * ["-", "--", "-.", ":"])
        if self.cycle_markers:
            cycles += cycler("marker", 2 * ["o", "s", "D", "^", "v", "x"])
        plt.rcParams["axes.prop_cycle"] = cycles

        # Ticks properties
        plt.rcParams["xtick.major.size"] = 4
        plt.rcParams["xtick.major.width"] = 0.8
        plt.rcParams["xtick.minor.size"] = 2
        plt.rcParams["xtick.minor.width"] = 0.5
        plt.rcParams["ytick.major.size"] = 4
        plt.rcParams["ytick.major.width"] = 0.8
        plt.rcParams["ytick.minor.size"] = 2
        plt.rcParams["ytick.minor.width"] = 0.5

        # Legend properties
        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.fontsize"] = 9

        # Figure properties
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["figure.edgecolor"] = "white"
        plt.rcParams["figure.dpi"] = self.dpi

        # Save properties
        plt.rcParams["savefig.dpi"] = self.dpi
        plt.rcParams["savefig.format"] = self.file_ext.strip(".")
        plt.rcParams["savefig.transparent"] = False

    def _get_axis_extent(self, ax: Axes, padding: float) -> transforms.Bbox:
        """Get the full bounding box of an axis including labels, ticks, and titles."""
        if self.fig is None or self.fig.canvas is None:
            raise RuntimeError("Figure is not initialized or canvas is unavailable.")
        self.fig.canvas.draw()
        elements = [ax, ax.xaxis.label, ax.yaxis.label, ax.title]
        bbox = transforms.Bbox.union([el.get_window_extent() for el in elements if el])
        return bbox.expanded(1.0 + padding, 1.0 + padding)

    def _save_subplot(
        self,
        ax: Axes,
        filename: str | Path,
        padding: float = 0.05,
        include_title: bool = True,
    ) -> None:
        """Save individual subplot with precise cropping."""
        try:
            if self.fig is None or self.fig.dpi_scale_trans is None:
                raise RuntimeError(
                    "Figure is not initialized or dpi_scale_trans is unavailable."
                )

            # Store original visibility of all axes
            all_axes = self.fig.axes
            original_visibility = [a.get_visible() for a in all_axes]

            # Hide all other axes so their labels don't affect the bbox
            for other_ax in all_axes:
                if other_ax is not ax:
                    other_ax.set_visible(False)

            original_title = ax.get_title()
            if not include_title:
                ax.set_title("")

            self.fig.canvas.draw()
            bbox = self._get_axis_extent(ax, padding).transformed(
                self.fig.dpi_scale_trans.inverted()
            )

            self.fig.savefig(
                filename,
                dpi=self.dpi,
                bbox_inches=bbox,
                format=self.file_ext.strip("."),
                transparent=True,
            )

            # Restore title
            if not include_title:
                ax.set_title(original_title)

            # Restore visibility
            for other_ax, vis in zip(all_axes, original_visibility):
                other_ax.set_visible(vis)

            print(f"Saved subplot to {filename}")

        except Exception as e:
            print(f"Error saving subplot {filename}: {e}")

    def set_figure_size(self, fig: Figure, n_rows: int, n_cols: int, horizontal: bool = False) -> None:
        """Set figure dimensions based on standard paper sizes."""
        paper_dimensions = {"A4": (8.27, 11.69), "A3": (11.69, 16.54)}
        width, height = paper_dimensions.get(
            self.paper_size, paper_dimensions["A4"]
        )  # default to A4
        if horizontal:
            width, height = height, width

        # Adjust for margins (1 inch total) and maintain aspect ratio
        margin = 0.5  # 0.5 inch margin on each side
        usable_width = width - 2 * margin
        subplot_width = usable_width / n_cols
        subplot_height = subplot_width * 0.75  # Adjusted for better LaTeX fit

        fig.set_size_inches(usable_width, subplot_height * n_rows)

    def create_figure(
        self, n_rows: int, n_cols: int, n_subplots: int, horizontal: bool = False, projection: str | None = None
    ) -> tuple[Figure, list[Axes]]:
        """Create a figure with subplots and apply formatting."""
        if n_subplots > n_rows * n_cols:
            raise ValueError("n_subplots cannot exceed n_rows * n_cols")

        # Apply custom style settings
        self._apply_custom_style()

        fig, axes_array = plt.subplots(n_rows, n_cols, squeeze=False, subplot_kw={'projection': projection})
        axes: list[Axes] = axes_array.flatten().tolist()  # pyright: ignore[reportAssignmentType]

        # Deactivate unused subplots
        for i in range(n_subplots, len(axes)):
            axes[i].axis("off")

        # Set figure size
        self.set_figure_size(fig, n_rows, n_cols, horizontal)

        # Apply styles to active subplots
        for ax in axes[:n_subplots]:
            sns.despine(ax=ax)

        # Store these for later use in save_figure
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_subplots = n_subplots
        self.fig = fig
        self.axes = axes[:n_subplots]

        return fig, axes[:n_subplots]

    def save_figure(self, filename: str = "figure", include_title: bool = True, subplot_settings: dict = {}) -> None:
        """Save the full figure and individual subplots."""
        # Ensure create_figure has been called
        if self.fig is None or self.axes is None:
            raise RuntimeError("Call create_figure before saving the figure.")

        # If path does not exist, create it
        if not self.output_dir.exists():
            print(f"Creating output directory: {self.output_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Apply tight layout before saving
        self.fig.tight_layout()

        # Save the full figure
        full_path = self.output_dir / f"{filename}{self.file_ext}"
        try:
            self.fig.savefig(
                full_path,
                dpi=self.dpi,
                bbox_inches="tight",
                format=self.file_ext.strip("."),
                transparent=True,
            )
            print(f"Saved full figure to {full_path}")
        except Exception as e:
            print(f"Error saving full figure: {e}")

        # Save each subplot separately
        for i, ax in enumerate(self.axes):
            subplot_path = (
                self.output_dir / f"{filename}_subplot_{i + 1}{self.file_ext}"
            )
            self._save_subplot(ax, subplot_path, include_title=include_title, **subplot_settings)