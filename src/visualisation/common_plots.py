"""
This file contains a class for drawing plots using matplotlib and seaborn. 
It provides methods for creating single and multiple bar plots with customizable aesthetics, labels, and formatting options.
"""

from matplotlib.ticker import FuncFormatter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.ticker as mticker
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class BarPlot:
    def __init__(self, df:pd.DataFrame):
        self.df = df

    @staticmethod
    def readable_numbers(x: float) -> str:
        """
        takes a large number and formats it into K,M to make it more readable

        Args
            x: float value to format

        Returns
            str: formatted string
        """
        if x >= 1e6:
            s = '{:1.2f}M'.format(x*1e-6)
        else:
            s = '{:1.2f}K'.format(x*1e-3)
        return s
    
    def set_locator(self, axes: Axes):
        """
        Format x, y ticklabels into more readable form

        Args
            axes: matplotlib axes to format x, y ticklabels
        """
        def is_number(x):
            x = x.replace('−', '-')
            try:
                float(x)
                return True
            except ValueError:
                return False
    
        x_ticklabels = [tick.get_text() for tick in axes.get_xticklabels()]
        y_ticklabels = [tick.get_text() for tick in axes.get_yticklabels()]
    
        formatter = FuncFormatter(lambda val, pos: self.readable_numbers(val))

        if is_number(x_ticklabels[0]) == True:
            axes.xaxis.set_major_formatter(formatter)
        if is_number(y_ticklabels[0]) == True:
            axes.yaxis.set_major_formatter(formatter)

    def single_plot(
        self,
        x_axis: str,
        y_axis: str,
        hue: str | None,
        kwarg: dict|None = None,
        readable_label: bool = False,
        ax: Axes = None
    ) -> Axes:
        """
        Draw a beautiful and readable bar plot.
    
        Args:
            x_axis: column for x-axis
            y_axis: column for y-axis
            hue: column to group bars (optional)
            readable_label: whether to display readable number labels
            ax: matplotlib axes to draw plot
            kwarg: other keyword arguments (e.g., from seaborn barplot)
        """
    
        # --- Setup Axes ---
        if ax is None:
            ax = plt.gca()
    
        # --- 🌟 Modern aesthetic theme ---
        palette = sns.color_palette("Spectral", n_colors=self.df[hue].nunique()) if hue else "viridis"
        # or: "rocket", "magma", "coolwarm", "crest", "flare", "viridis"

        if kwarg is None:
            kwarg = {}
        # --- Draw the barplot ---
        bar = sns.barplot(
            data=self.df,
            x=x_axis,
            y=y_axis,
            hue=hue,
            ax=ax,
            # palette=palette,
            edgecolor="black",
            linewidth=1.2,
            **kwarg
        )
    
        # --- 🧠 Title & Labels ---
        ax.set_title(
            f'{y_axis} by {x_axis}',
            fontdict={
                'fontsize': 16,
                'fontweight': 'regular',
                # 'fontname': 'Times New Roman'
            },
            pad=15
        )
        ax.set_xlabel(x_axis, fontsize=12, labelpad=10)
        ax.set_ylabel(y_axis, fontsize=12, labelpad=10)
    
        # --- ✨ Ticks & Grid ---
        ax.tick_params(axis='x', labelsize=10, rotation=0)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
    
        # --- 🎨 Legend ---
        if hue is not None:
            ax.legend(
                title=hue,
                title_fontsize=11,
                fontsize=10,
                loc='best',
                frameon=False
            )
    
        # --- 💬 Bar labels ---
        if readable_label:
            for container in bar.containers:
                labels = [p.get_height() for p in container]
                bar.bar_label(
                    container,
                    labels=labels,
                    label_type='edge',
                    fontsize=9,
                    # fontname='Times New Roman',
                    color='black',
                    padding=2
                )
            self.set_locator(bar)
    
        # # --- 🌤️ Tighten layout ---
        # plt.tight_layout()
    
        return bar

    def multiple_plots(self, x_axis_list: list[str], y_axis_list: list[str], hue_list: list[str|None], kwarg: dict,
                       plots_each_row: int, readable_label: bool = False) -> tuple[Figure, np.array[[Axes]]]:
        """
        Draw multiple plots

        Args
            Same as single_plot
            y_axis_list: list of columns for y axis
            plots_each_row: number of plots for each row in matplotlib figure
            kwarg: other keyword arguments if needed.
        """
        
        combination = list(zip(x_axis_list, y_axis_list, hue_list))
        
        length = len(combination)
        number_rows = math.ceil(length / plots_each_row)
    
        fig, axes = plt.subplots(nrows=number_rows, ncols=plots_each_row, figsize=(16, 8),
                                sharey=True)
        
        axes = np.array(axes).reshape(number_rows, plots_each_row)
    
        coordinates = list(itertools.product(range(number_rows), range(plots_each_row)))
    
        for coor, com in zip(coordinates, combination):
            x, y, hue = com
            ax = axes[coor]
            self.single_plot(
                x_axis=x,
                y_axis=y,
                hue=hue,
                readable_label=readable_label,
                kwarg=kwarg,
                ax=ax
            )
    
        # hide any unused axes
        for coor in coordinates[length:]:
            fig.delaxes(axes[coor])

        return fig, axes


class BoxPlot:
    """
    Create single and multiple box plots
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def readable_numbers(x: float) -> str:
        """
        takes a large number and formats it into K,M to make it more readable

        Args
            x: float value to format

        Returns
            str: formatted string
        """
        if x >= 1e6:
            s = '{:1.2f}M'.format(x*1e-6)
        else:
            s = '{:1.2f}K'.format(x*1e-3)
        return s

    def set_locator(self, axes: Axes):
        """
        Format x, y ticklabels into more readable form

        Args
            axes: matplotlib axes to format x, y ticklabels
        """
        def is_number(x):
            x = x.replace('−', '-')
            try:
                float(x)
                return True
            except ValueError:
                return False
    
        x_ticklabels = [tick.get_text() for tick in axes.get_xticklabels()]
        y_ticklabels = [tick.get_text() for tick in axes.get_yticklabels()]

        formatter = FuncFormatter(lambda val, pos: self.readable_numbers(val))

        if is_number(x_ticklabels[0]) == True:
            axes.xaxis.set_major_formatter(formatter)
        if is_number(y_ticklabels[0]) == True:
            axes.yaxis.set_major_formatter(formatter)

    def single_plot(
        self,
        x_axis: str,
        y_axis: str | None,
        hue: str | None,
        kwarg: dict | None = None,
        readable_label: bool = False,
        ax: Axes = None
    ) -> Axes:
        """
        Draw a beautiful and readable Box Plot.
    
        Args:
            x_axis: column needed
            y_axis: column needed
            hue: column needed
            dimension: '1d' for single variable, '2d' for grouped boxplot
            readable_label: whether to format x/y tick labels for readability
            ax: matplotlib axes to draw plot
            kwarg: additional seaborn arguments
        """
        
        if ax is None:
            ax = plt.gca()
        if kwarg is None:
            kwarg = {}
            
        # --- Styling palette and defaults ---
        sns.set_style("whitegrid")  # clean background
        palette = kwarg.pop("palette", "Set2") if kwarg else "Set2"
            
        if hue is None:
            palette=None
        
        box = sns.boxplot(
            data=self.df,
            x=x_axis,
            y=y_axis,
            hue=hue,
            ax=ax,
            palette=palette,
            width=0.6,
            fliersize=3,
            linewidth=1.2,
            boxprops={"alpha": 0.8},
            medianprops={"color": "black", "linewidth": 1},
            whiskerprops={"linewidth": 1},
            capprops={"linewidth": 1},
            **kwarg
        )
        
        if y_axis is None:
            y_axis = ""
        
        ax.set_xlabel(x_axis, fontsize=13, labelpad=8)
        ax.set_ylabel(y_axis, fontsize=13, labelpad=8)

        if hue is not None and y_axis != "":
            ax.legend(
                title=hue if hue else "",
                frameon=True,
                fancybox=True,
                framealpha=0.8,
                fontsize=11,
                title_fontsize=12,
                loc="best",
            )
    
        # --- Titles & Ticks ---
        ax.set_title(
            f"Box Plot",
            fontdict={"fontsize": 12, "fontweight": "regular"}, # "fontname": "Times New Roman"
            pad=15
        )
        ax.tick_params(axis="x", labelrotation=0, labelsize=11)
        ax.tick_params(axis="y", labelrotation=0, labelsize=11)
    
        # --- Optional: readable tick labels ---
        if readable_label:
            self.set_locator(ax)
    
        # --- Clean up look ---
        sns.despine(ax=ax, top=True, right=True)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
    
        return box
    
    def multiple_plots(self, y_axis_list: list[str|None], x_axis_list: list[str], kwarg: dict|None,
                       hue_list: list[str|None], plots_each_row: int, 
                       readable_label: bool = False) -> tuple[Figure, np.array[[Axes]]]:
        """
        Draw multiple box plots
        
        Args
            Same as single_plot
            y_axis_list: list of columns for y axis
            plots_each_row: number of plots for each row in matplotlib figure
        """
        combination = list(zip(x_axis_list, y_axis_list, hue_list))
        
        length = len(combination)
        number_rows = math.ceil(length / plots_each_row)
    
        fig, axes = plt.subplots(nrows=number_rows, ncols=plots_each_row, figsize=(16, 8),
                                sharey=True) # same y_axis scale
        axes = np.array(axes).reshape(number_rows, plots_each_row)
    
        coordinates = list(itertools.product(range(number_rows), range(plots_each_row)))
    
        for coor, com in zip(coordinates, combination):
            x, y, hue = com
            ax = axes[coor]
            self.single_plot(
                x_axis=x,
                y_axis=y,
                hue=hue,
                readable_label=readable_label,
                kwarg=kwarg,
                ax=ax
            )
    
        # hide any unused axes
        for coor in coordinates[length:]:
            fig.delaxes(axes[coor])
    
        fig.tight_layout()

        return fig, axes


class KDEPlot:
    """
    Create single and multiple box plots
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def readable_numbers(x: float) -> str:
        """
        takes a large number and formats it into K,M to make it more readable

        Args
            x: float value to format

        Returns
            str: formatted string
        """
        if x >= 1e6:
            s = '{:1.1f}M'.format(x*1e-6)
        elif x >= 1e3:
            s = '{:1.1f}K'.format(x*1e-3)
        else:
            s = x
        return s

    def set_locator(self, axes: Axes):
        """
        Format x, y ticklabels into more readable form

        Args
            axes: matplotlib axes to format x, y ticklabels
        """
        def is_number(x):
            x = x.replace('−', '-')
            try:
                float(x)
                return True
            except ValueError:
                return False
    
        x_ticklabels = [tick.get_text() for tick in axes.get_xticklabels()]
        y_ticklabels = [tick.get_text() for tick in axes.get_yticklabels()]

        formatter = FuncFormatter(lambda val, pos: self.readable_numbers(val))

        if is_number(x_ticklabels[1]) == True:
            axes.xaxis.set_major_formatter(formatter)
        if is_number(y_ticklabels[1]) == True:
            axes.yaxis.set_major_formatter(formatter)

    def single_plot(
        self,
        x_axis: str,
        y_axis: str | None,
        hue: str | None,
        kwarg: dict | None = None,
        readable_label: bool = False,
        ax: Axes = None
    ) -> Axes:
        """
        Draw a beautiful and readable Box Plot.
    
        Args:
            x_axis: column needed
            y_axis: column needed
            hue: column needed
            dimension: '1d' for single variable, '2d' for grouped boxplot
            readable_label: whether to format x/y tick labels for readability
            ax: matplotlib axes to draw plot
            kwarg: additional seaborn arguments
        """
        
        if ax is None:
            ax = plt.gca()
        if kwarg is None:
            kwarg = {}
            
        # --- Styling palette and defaults ---
        sns.set_style("whitegrid")  # clean background
        palette = sns.color_palette('pastel')
        # or: "rocket", "magma", "coolwarm", "crest", "flare", "viridis"

        if hue is None:
            palette = None
        
        kde = sns.kdeplot(
            data=self.df,
            x=x_axis,
            y=y_axis,
            hue=hue,
            ax=ax,
            palette=palette,
            **kwarg
        )

        
        ax.set_xlabel(x_axis, fontsize=13, labelpad=8)
        ax.set_ylabel(y_axis, fontsize=13, labelpad=8)

        # if hue is not None:
        #     ax.legend(
        #         title=hue if hue else "",
        #         frameon=True,
        #         fancybox=True,
        #         framealpha=0.8,
        #         fontsize=11,
        #         title_fontsize=12,
        #         loc="best",
        #     )
    
        # --- Titles & Ticks ---
        ax.set_title(
            f"KDE Plot",
            fontdict={"fontsize": 12, "fontweight": "regular", 
                      # "fontname": "Times New Roman"
                     },
            pad=15
        )
        ax.tick_params(axis="x", labelrotation=0, labelsize=11)
        ax.tick_params(axis="y", labelrotation=0, labelsize=11)
    
        # --- Optional: readable tick labels ---
        if readable_label:
            self.set_locator(ax)
    
        # --- Clean up look ---
        sns.despine(ax=ax, top=True, right=True)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
    
        return kde
    
    def multiple_plots(self, y_axis_list: list[str|None], x_axis_list: list[str], kwarg: dict|None,
                       hue_list: list[str|None], plots_each_row: int, 
                       readable_label: bool = False) -> tuple[Figure, np.array[[Axes]]]:
        """
        Draw multiple box plots
        
        Args
            Same as single_plot
            y_axis_list: list of columns for y axis
            plots_each_row: number of plots for each row in matplotlib figure
        """
        combination = list(zip(x_axis_list, y_axis_list, hue_list))
        
        length = len(combination)
        number_rows = math.ceil(length / plots_each_row)
    
        fig, axes = plt.subplots(nrows=number_rows, ncols=plots_each_row, figsize=(16, 8),
                                ) # sharey=True: same y_axis scale
        axes = np.array(axes).reshape(number_rows, plots_each_row)
    
        coordinates = list(itertools.product(range(number_rows), range(plots_each_row)))
    
        for coor, com in zip(coordinates, combination):
            x, y, hue = com
            ax = axes[coor]
            self.single_plot(
                x_axis=x,
                y_axis=y,
                hue=hue,
                readable_label=readable_label,
                kwarg=kwarg,
                ax=ax
            )
    
        # hide any unused axes
        for coor in coordinates[length:]:
            fig.delaxes(axes[coor])
    
        fig.tight_layout()

        return fig, axes


class CountPlot:
    def __init__(self, df:pd.DataFrame):
        self.df = df

    @staticmethod
    def readable_numbers(x: float) -> str:
        """
        takes a large number and formats it into K,M to make it more readable

        Args
            x: float value to format

        Returns
            str: formatted string
        """
        if x >= 1e6:
            s = '{:1.2f}M'.format(x*1e-6)
        else:
            s = '{:1.2f}K'.format(x*1e-3)
        return s
    
    def set_locator(self, axes: Axes):
        """
        Format x, y ticklabels into more readable form

        Args
            axes: matplotlib axes to format x, y ticklabels
        """
        def is_number(x):
            x = x.replace('−', '-')
            try:
                float(x)
                return True
            except ValueError:
                return False
    
        x_ticklabels = [tick.get_text() for tick in axes.get_xticklabels()]
        y_ticklabels = [tick.get_text() for tick in axes.get_yticklabels()]
    
        formatter = FuncFormatter(lambda val, pos: self.readable_numbers(val))

        if is_number(x_ticklabels[0]) == True:
            axes.xaxis.set_major_formatter(formatter)
        if is_number(y_ticklabels[0]) == True:
            axes.yaxis.set_major_formatter(formatter)

    def single_plot(
        self,
        x_axis: str,
        hue: str | None,
        kwarg: dict|None = None,
        readable_label: bool = False,
        ax: Axes = None
    ) -> Axes:
        """
        Draw a beautiful and readable bar plot.
    
        Args:
            x_axis: column for x-axis
            y_axis: column for y-axis
            hue: column to group bars (optional)
            readable_label: whether to display readable number labels
            ax: matplotlib axes to draw plot
            kwarg: other keyword arguments (e.g., from seaborn barplot)
        """
    
        # --- Setup Axes ---
        if ax is None:
            ax = plt.gca()
        if kwarg is None:
            kwarg = {}

        if hue is None:
            palette = None
        else:
            # --- 🌟 Modern aesthetic theme ---
            palette = sns.color_palette("Spectral", n_colors=self.df[hue].nunique()) if hue else "viridis"
            # or: "rocket", "magma", "coolwarm", "crest", "flare", "viridis"
        
        # --- Draw the barplot ---
        count = sns.countplot(
            data=self.df,
            x=x_axis,
            hue=hue,
            ax=ax,
            palette=palette,
            edgecolor="black",
            linewidth=1.2,
            **kwarg
        )
    
        # --- 🧠 Title & Labels ---
        ax.set_title(
            f'CountPlot',
            fontdict={
                'fontsize': 12,
                'fontweight': 'regular',
                # 'fontname': 'Times New Roman'
            },
            pad=5
        )
        ax.set_xlabel(x_axis, fontsize=12, labelpad=8)
        ax.set_ylabel('Count', fontsize=12, labelpad=8)
    
        # --- ✨ Ticks & Grid ---
        ax.tick_params(axis='x', labelsize=10, rotation=0)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
    
        # --- 🎨 Legend ---
        if hue is not None:
            ax.legend(
                title=hue,
                title_fontsize=11,
                fontsize=10,
                loc='best',
                frameon=False
            )
    
        # --- 💬 Bar labels ---
        if readable_label:
            for container in count.containers:
                labels = [self.readable_numbers(p.get_height()) for p in container]
                count.bar_label(
                    container,
                    labels=labels,
                    label_type='edge',
                    fontsize=9,
                    fontname='Times New Roman',
                    color='black',
                    padding=2
                )
            self.set_locator(count)
    
        # # --- 🌤️ Tighten layout ---
        # plt.tight_layout()
    
        return count

    def multiple_plots(self, x_axis_list: list[str], hue_list: list[str|None], plots_each_row: int,
                       kwarg: dict|None = None, readable_label: bool = False) -> tuple[Figure, np.array[[Axes]]]:
        """
        Draw multiple plots

        Args
            Same as single_plot
            x_axis_list: list of columns for x axis
            plots_each_row: number of plots for each row in matplotlib figure
            kwarg: other keyword arguments if needed.
        """
        
        combination = list(zip(x_axis_list, hue_list))
        
        length = len(combination)
        number_rows = math.ceil(length / plots_each_row)
    
        fig, axes = plt.subplots(nrows=number_rows, ncols=plots_each_row, figsize=(20, 16),
                                sharey=False) # same y_axis scale
        axes = np.array(axes).reshape(number_rows, plots_each_row)
    
        coordinates = list(itertools.product(range(number_rows), range(plots_each_row)))
    
        for coor, com in zip(coordinates, combination):
            x, hue = com
            ax = axes[coor]
            self.single_plot(
                x_axis=x,
                hue=hue,
                readable_label=readable_label,
                kwarg=kwarg,
                ax=ax,
            )
    
        # hide any unused axes
        for coor in coordinates[length:]:
            fig.delaxes(axes[coor])

        fig.tight_layout(pad=2.5)

        return fig, axes


class PieChart:
    def __init__(self, df:pd.DataFrame):
        self.df = df

    def single_plot(
        self,
        column: str, 
        kwarg: dict=None,
        ax: Axes = None
    ) -> Axes:
        """
        Draw a beautiful and readable bar plot.
    
        Args:
            x_axis: column for x-axis
            y_axis: column for y-axis
            ax: matplotlib axes to draw plot
            kwarg: other keyword arguments (e.g., from plot)
        """
    
        # --- Setup Axes ---
        if ax is None:
            ax = plt.gca()
    
        # --- 🌟 Modern aesthetic theme ---
        palette = sns.color_palette('pastel')
        # or: "rocket", "magma", "coolwarm", "crest", "flare", "viridis"
        
        # --- Draw the barplot ---
        series = self.df[column].value_counts(normalize=True)
        data=series.values
        labels=series.index

        if kwarg is None:
            kwarg = {}

        wedges, texts, autotexts = ax.pie(
            x=data,
            labels=labels,
            colors=palette,
            startangle=90,
            autopct='%1.1f%%',
            wedgeprops=dict(edgecolor='black'), # width=0.5
            **kwarg
        )
    
        # --- 🧠 Title & Labels ---
        ax.set_title(
            f'Pie Chart for {column}',
            fontdict={
                'fontsize': 12,
                # 'fontweight': 'bold',
                # 'fontname': 'Times New Roman'
            },
            pad=4
        )
    
        # ax.legend(labels, title="Classes", loc="best", 
        #           bbox_to_anchor=(1, 0, 0.5, 1), frameon=False)

        # # --- 🌤️ Tighten layout ---
        # plt.tight_layout()
    
        return ax

    def multiple_plots(self, column_list: list[str], plots_each_row: int,
                   kwarg: dict = None) -> tuple[Figure, np.array[[Axes]]]:
        combination = column_list
        length = len(combination)
        number_rows = math.ceil(length / plots_each_row)
    
        # Scale figure size with number of rows & columns
        fig_width  = 5.5 * plots_each_row          # ~5.5 inch per column
        fig_height = 4.5 * number_rows             # ~4.5 inch per row
        fig, axes = plt.subplots(
            nrows=number_rows,
            ncols=plots_each_row,
            figsize=(fig_width, fig_height),
            sharey=False
        )
        axes = np.array(axes).reshape(number_rows, plots_each_row)
    
        coordinates = list(itertools.product(range(number_rows), range(plots_each_row)))
    
        for coor, com in zip(coordinates, combination):
            column = com
            ax = axes[coor]
            self.single_plot(column=column, kwarg=kwarg, ax=ax)
    
        # Hide unused axes
        for coor in coordinates[length:]:
            fig.delaxes(axes[coor])
    
        # Important: give the pies room
        fig.tight_layout(pad=2.5)          # or try pad=1.5 ~ 3.0
        # Alternative if tight_layout is not enough:
        # fig.subplots_adjust(hspace=0.45, wspace=0.25)
    
        return fig, axes