import threading

from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from transcription import TranscriptEntry

MAX_VISIBLE_ENTRIES = 60


def _build_header(status: str, model_size: str, entry_count: int) -> Panel:
    status_color = "green" if status == "Listening" else "yellow"
    left = Text.assemble(
        ("  LIVE TRANSCRIPTION", "bold cyan"),
        ("  |  ", "dim"),
        ("model: ", "dim"),
        (model_size, "cyan"),
    )
    right = Text.assemble(
        ("status: ", "dim"),
        (status, status_color),
        ("  |  ", "dim"),
        ("entries: ", "dim"),
        (str(entry_count), "cyan"),
        ("  ", ""),
    )
    # Pad left to push right to the far side
    left.justify = "left"
    right.justify = "right"
    row = Columns([left, right], expand=True)
    return Panel(row, style="bold", border_style="cyan", height=3)


def _build_transcript(entries: list[TranscriptEntry]) -> Panel:
    if not entries:
        placeholder = Text("\n  Waiting for speech...", style="dim italic white")
        return Panel(placeholder, title="Transcript", border_style="cyan", padding=(0, 1))

    visible = entries[-MAX_VISIBLE_ENTRIES:]

    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim cyan", no_wrap=True, min_width=10)
    table.add_column(style="white", overflow="fold")

    for entry in visible:
        table.add_row(f"[{entry.timestamp}]", entry.text)

    return Panel(table, title="Transcript", border_style="cyan", padding=(0, 1))


def _build_footer() -> Panel:
    hint = Text("  Press Ctrl+C to stop and save transcript", style="dim")
    return Panel(hint, style="dim", border_style="dim", height=3)


def render(
    transcript_store: list,
    transcript_lock: threading.Lock,
    status: str,
    model_size: str,
) -> Group:
    with transcript_lock:
        entries = list(transcript_store)

    header = _build_header(status, model_size, len(entries))
    transcript = _build_transcript(entries)
    footer = _build_footer()

    return Group(header, transcript, footer)
