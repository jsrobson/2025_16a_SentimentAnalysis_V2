import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
import pandas as pd

from user_interface import ProgressPopup
from utils import CSVLoader
from processor import Parser


class UserInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CSV Topic Processor")

        # vars
        self.csv_path = tk.StringVar()
        self.column_selected = tk.StringVar()
        self.save_path= tk.StringVar()
        self.topics_csv_path = tk.StringVar()
        self.topics_column_selected = tk.StringVar()

        # tools
        self.df_in = None
        self.seeds = None
        self.parser = None

        self._build_ui()

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Allow column 1 to expand (right column)
        frame.columnconfigure(1, weight=1)

        # ---------- Load CSV ----------
        tk.Label(frame, text="Load CSV:").grid(row=0, column=0, sticky="w",
                                               pady=5)
        tk.Entry(frame, textvariable=self.csv_path).grid(row=0, column=1,
                                                         sticky="ew", padx=5)
        tk.Button(frame, text="Browse...", command=self._load_csv).grid(row=0,
                                                                        column=2,
                                                                        padx=5)

        # Column selection
        tk.Label(frame, text="Select Column:").grid(row=1, column=0,
                                                    sticky="w", pady=5)
        self.column_combobox = ttk.Combobox(frame,
                                            textvariable=self.column_selected,
                                            state="readonly")
        self.column_combobox.grid(row=1, column=1, sticky="ew", padx=5)

        # Separator
        ttk.Separator(frame, orient="horizontal").grid(row=2, column=0,
                                                       columnspan=3,
                                                       sticky="ew", pady=10)

        # ---------- Topics CSV ----------
        tk.Label(frame, text="OPTIONAL: Seed Topics (CSV):").grid(row=3,
                                                                  column=0,
                                                                  sticky="w",
                                                                  pady=5)
        tk.Entry(frame, textvariable=self.topics_csv_path).grid(row=3,
                                                                column=1,
                                                                sticky="ew",
                                                                padx=5)
        tk.Button(frame, text="Browse...", command=self._load_topics_csv).grid(
            row=3, column=2, padx=5)

        # Column selection for topics
        tk.Label(frame, text="Select Column:").grid(row=4, column=0,
                                                    sticky="w", pady=5)
        self.topics_column_combobox = ttk.Combobox(frame,
                                            textvariable=self.topics_column_selected,
                                            state="readonly")
        self.topics_column_combobox.grid(row=4, column=1, sticky="ew", padx=5)

        # Separator
        ttk.Separator(frame, orient="horizontal").grid(row=5, column=0,
                                                       columnspan=3,
                                                       sticky="ew", pady=10)

        # ---------- Save CSV ----------
        tk.Label(frame, text="Save CSV:").grid(row=6, column=0, sticky="w",
                                               pady=5)
        tk.Entry(frame, textvariable=self.save_path).grid(row=6, column=1,
                                                          sticky="ew", padx=5)
        tk.Button(frame, text="Browse...",
                  command=self._browse_save_location).grid(row=6, column=2,
                                                           padx=5)

        # Separator
        ttk.Separator(frame, orient="horizontal").grid(row=7, column=0,
                                                       columnspan=3,
                                                       sticky="ew", pady=10)

        # ---------- Run Button ----------
        run_font = tkfont.Font(weight="bold", size=14)

        tk.Button(
            frame,
            text="RUN",
            command=self.run_processing,
            bg="white",
            fg="green",
            font=run_font,
            height=2
        ).grid(row=8, column=0, columnspan=3, sticky="ew", pady=0)

        # ---------- Reset Button ----------
        run_font = tkfont.Font(weight="bold", size=14)

        tk.Button(
            frame,
            text="RESET",
            command=self._reset,
            bg="white",
            fg="red",
            font=run_font,
            height=2
        ).grid(row=9, column=0, columnspan=3, sticky="ew", pady=0)


    def _load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.csv_path.set(path)
            self.df_in = CSVLoader(self.csv_path.get()).load()
            self.column_combobox['values'] = list(self.df_in.columns)
            if self.df_in.columns.any():
                self.column_combobox.current(0)

    def _browse_save_location(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv")])
        if path:
            self.save_path.set(path)

    def _load_topics_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.topics_csv_path.set(path)
            df_topics = CSVLoader(self.topics_csv_path.get()).load()

            # update combobox values
            self.topics_column_combobox['values'] = list(df_topics.columns)

            # auto-select the first column if it exists
            if len(df_topics.columns) > 0:
                self.topics_column_combobox.current(0)
                self._on_topics_column_selected(df_topics)  # load seeds
                # immediately

    def run_processing(self):
        if not self.csv_path.get():
            messagebox.showerror("Error", "Please load a CSV file")
            return
        if not self.column_selected.get():
            messagebox.showerror("Error", "Please select a column")
            return
        if not self.save_path.get():
            messagebox.showerror("Error", "Please select a save location")
            return

        self.parser = Parser(self.df_in, self.column_selected.get(), self.seeds)
        progress = ProgressPopup(self.root, message="Initializing tasks...")

        # Run long task in background thread
        import threading
        def background_task():
            try:
                # Step 1: Text clusters
                self.root.after(0, lambda: progress.update_message(
                    "Building text clusters..."))
                self.root.after(0, lambda: progress.log(
                    "Preprocessing text for clustering..."))
                self.parser.pre_process_ml()
                self.root.after(0, lambda: progress.log(
                    "Text clustering complete."))

                # Step 2: Topics & subtopics
                self.root.after(0, lambda: progress.update_message(
                    "Identifying topics and subtopics..."))
                self.root.after(0, lambda: progress.log(
                    "Building hierarchical data structures..."))
                self.parser.build_data_structures()
                self.root.after(0, lambda: progress.log(
                    "Topics & subtopics identified."))

                # Step 3: LLM processing
                self.root.after(0, lambda: progress.update_message(
                    "Running LLM model..."))
                self.root.after(0, lambda: progress.log(
                    "Sending prompts to Gemma 3 LLM..."))
                self.parser.process_llm()
                self.root.after(0, lambda: progress.log(
                    "LLM processing complete."))

                # Step 4: Save results
                self.root.after(0, lambda: progress.update_message(
                    "Saving results..."))
                self.root.after(0, lambda: progress.log(
                    f"Saving results to {self.save_path.get()}..."))
                self.parser.save(self.save_path.get())
                self.root.after(0, lambda: progress.log(
                    "Results saved successfully."))

            finally:
                self.root.after(0, lambda: progress.close())
        threading.Thread(target=background_task, daemon=True).start()

    def run(self):
        self.root.mainloop()

    def _on_topics_column_selected(self, topics: pd.DataFrame, event=None):
        column = self.topics_column_combobox.get()
        if not column:
            return

        col_series = topics[column]

        # Convert to list, dropping NaN
        self.seeds = col_series.dropna().astype(str).tolist()

        print("Seeds loaded:", self.seeds)

    def _reset(self):
        for var in [
            self.csv_path,
            self.column_selected,
            self.save_path,
            self.topics_csv_path,
            self.topics_column_selected,
        ]:
            var.set("")

        # tools
        self.df_in = None
        self.seeds = None
        self.parser = None
