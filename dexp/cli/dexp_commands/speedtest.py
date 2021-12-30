import click

from dexp.utils.speed_test import perform_speed_test


@click.command()
def speedtest():
    """Estimates storage medium speed."""

    perform_speed_test()
