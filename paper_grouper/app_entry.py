"""Canonical GUI entrypoint for Paper Grouper."""

from paper_grouper.ui.main_window import main as gui_main


def main() -> None:
    """Run the official GUI entrypoint."""
    gui_main()


if __name__ == "__main__":
    main()
