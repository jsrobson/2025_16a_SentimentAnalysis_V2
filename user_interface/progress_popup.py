import tkinter as tk
from tkinter import ttk, scrolledtext
import tkinter.font as tkfont

TITLE = "Processing..."
MSG = "Please wait..."

class ProgressPopup:
    def __init__(self, parent, title=TITLE, message=MSG):
        self.top = tk.Toplevel(parent)
        # initialize display options
        self._initialize_options(parent, title)
        self._build_popup_ui(message)


    def _build_popup_ui(self, message: str):
        # Font for the status label
        self.status_font = tkfont.Font(weight="bold", size=12)

        # Status label
        self.label = ttk.Label(self.top, text=message, font=self.status_font)
        self.label.pack(pady=(10, 5), padx=10)

        # Indeterminate progress bar
        self.progress = ttk.Progressbar(self.top, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=5)
        self.progress.start(10)

        # Scrollable log area
        self.log_text = scrolledtext.ScrolledText(
            self.top,
            height=15,
            state="disabled",
            wrap="word",
            font=("Consolas", 10)
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        # Close button (initially hidden)
        self.close_btn = tk.Button(self.top, text="Close",
                                   command=self.top.destroy, state="disabled")
        self.close_btn.pack(pady=(0, 10))

    def _initialize_options(self, parent, title: str):
        self.top.title(title)
        self.top.geometry("500x300")
        self.top.resizable(True, True)  # allow resizing
        self.top.transient(parent)
        self.top.grab_set()  # modal
        # Prevent window from being closed manually
        self.top.protocol("WM_DELETE_WINDOW", lambda: None)

    def update_message(self, message):
        self.label.config(text=message)

    def log(self, message):
        """Append a message to the log window and auto-scroll."""
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")  # auto-scroll to latest
        self.log_text.config(state="disabled")

    def close(self):
        """Stop the progress bar, grey out the log, enable the close button."""
        self.progress.stop()
        self.progress.destroy()
        self.label.config(text="Done")
        # Grey out log to indicate finished
        self.log_text.config(state="normal", fg="grey")
        # Enable the close button
        self.close_btn.config(state="normal")
