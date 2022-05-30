"""
Plotting functions for the GPEs.
"""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_top_ents(top_ents, output_dir, group_type="Real"):
    """Finds the most common entities and olots them in a horizontal bar chart"""
    plot_title = f"Most Mentioned {group_type} News GPEs"
    sns.barplot(data=top_ents, y="Entity", x="Count", orient="h", color="#29C5F6").set(
        title=plot_title
    )
    plt.savefig(str(output_dir / f"{group_type}_top_ents.png"))
