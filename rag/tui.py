from textual import events
from textual.app import App, ComposeResult
from textual.containers import Vertical, Container, VerticalScroll
from textual.widgets import Input, Label, Markdown, Static, TabbedContent, TabPane

JESSICA = """
# Lady Jessica

Bene Gesserit and concubine of Leto, and mother of Paul and Alia.
"""

PAUL = """
# Paul Atreides

Son of Leto and Jessica.
"""

TEXT = """\
Docking a widget removes it from the layout and fixes its position, aligned to either the top, right, bottom, or left edges of a container.

Docked widgets will not scroll out of view, making them ideal for sticky headers, footers, and sidebars.

"""

class TabbedApp(App):
    """An example of tabbed content."""

    CSS_PATH = "static/styles.tcss"

    BINDINGS = [
        ("n", "show_tab('rag')", "Rag"),
        ("e", "show_tab('settings')", "Settings"),
    ]

    def compose(self) -> ComposeResult:
        """Compose app with tabbed content."""
        # Add the TabbedContent widget
        with Container(id="rag"):
            with VerticalScroll(id="references"):
                yield Static("test3", classes="context")
            with VerticalScroll(id="history"):
                yield Static(TEXT * 10, classes="output")
            with Vertical(id="chat"):
                yield Static("test2", classes="input")
        # with TabbedContent(initial="rag"):
        #     with TabPane("RAG", id="rag"):  # First tab
        #         # yield Input(placeholder=">>", id="chat")
        #         yield Static("test1", classes="chat")
        #         yield Static("test", classes="context")
            # with TabPane("Settings", id="settings"):
            #     yield Markdown(JESSICA)
            #     with TabbedContent("Paul", "Alia"):
            #         yield TabPane("Paul", Label("First child"))
            #         yield TabPane("Alia", Label("Second child"))

    def action_show_tab(self, tab: str) -> None:
        """Switch to a new tab."""
        self.get_child_by_type(TabbedContent).active = tab

    def on_key(self, event: events.Key) -> None:
        if event.key == "s":
            self.action_show_tab("settings")
        if event.key == "r":
            self.action_show_tab("rag")
        if event.key == "q":
            self.exit()


if __name__ == "__main__":
    app = TabbedApp()
    app.run()
