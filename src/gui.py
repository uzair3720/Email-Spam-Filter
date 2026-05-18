import os
import re
import tkinter as tk
from tkinter import ttk, messagebox
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "spam_detector_model.pkl")
VECT_PATH  = os.path.join(ROOT, "tfidf_vectorizer.pkl")


def clean(text):
    """Must match the cleaning used in train_large.py exactly."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Theme
BG       = "#0F172A"
PANEL    = "#1E293B"
ACCENT   = "#3B82F6"
ACCENT_H = "#2563EB"
DANGER   = "#EF4444"
DANGER_H = "#DC2626"
TEXT     = "#E2E8F0"
MUTED    = "#94A3B8"
OK       = "#22C55E"
WARN     = "#F59E0B"


class SpamFilterAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Filter AI")
        self.root.geometry("720x640")
        self.root.configure(bg=BG)
        self.root.minsize(600, 560)

        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECT_PATH)

        # Header
        header = tk.Frame(root, bg=BG, pady=20)
        header.pack(fill="x", padx=24)
        tk.Label(header, text="Spam Filter AI",
                 font=("Helvetica", 26, "bold"), fg=TEXT, bg=BG).pack(anchor="w")
        tk.Label(header, text="Paste any email below and check whether it's spam.",
                 font=("Helvetica", 11), fg=MUTED, bg=BG).pack(anchor="w", pady=(4, 0))

        # Input panel
        panel = tk.Frame(root, bg=PANEL, padx=16, pady=16,
                         highlightthickness=1, highlightbackground="#334155")
        panel.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        tk.Label(panel, text="Email content",
                 font=("Helvetica", 11, "bold"), fg=MUTED, bg=PANEL).pack(anchor="w")

        self.email_text = tk.Text(
            panel, height=14, wrap="word",
            bg="#0B1220", fg=TEXT, insertbackground=TEXT,
            font=("Consolas", 11), relief="flat", padx=12, pady=10,
        )
        self.email_text.pack(fill="both", expand=True, pady=(8, 0))

        # Buttons
        button_row = tk.Frame(root, bg=BG)
        button_row.pack(fill="x", padx=24)

        self.submit_btn = self._button(button_row, "Check Email",
                                       self.process_email, ACCENT, ACCENT_H)
        self.submit_btn.pack(side="left")

        self.clear_btn = self._button(button_row, "Clear",
                                      self.delete_mail, DANGER, DANGER_H)
        self.clear_btn.pack(side="left", padx=(10, 0))

        # Result card
        self.result_card = tk.Frame(root, bg=PANEL, padx=20, pady=18,
                                    highlightthickness=1, highlightbackground="#334155")
        self.result_card.pack(fill="x", padx=24, pady=(16, 8))

        self.result_label = tk.Label(
            self.result_card, text="Awaiting input",
            font=("Helvetica", 16, "bold"), fg=TEXT, bg=PANEL,
        )
        self.result_label.pack(anchor="w")

        self.confidence_label = tk.Label(
            self.result_card, text="Paste an email and click Check.",
            font=("Helvetica", 11), fg=MUTED, bg=PANEL,
        )
        self.confidence_label.pack(anchor="w", pady=(4, 8))

        self.bar_bg = tk.Frame(self.result_card, bg="#0B1220", height=8)
        self.bar_bg.pack(fill="x")
        self.bar_fill = tk.Frame(self.bar_bg, bg=ACCENT, height=8, width=0)
        self.bar_fill.place(x=0, y=0, relheight=1)

        # Footer
        tk.Label(root, text="© 2024 Spam Filter AI",
                 font=("Helvetica", 9), fg=MUTED, bg=BG).pack(side="bottom", pady=8)

    def _button(self, parent, text, cmd, color, hover):
        b = tk.Button(
            parent, text=text, command=cmd,
            font=("Helvetica", 12, "bold"), fg="#FFFFFF", bg=color,
            activebackground=hover, activeforeground="#FFFFFF",
            relief="flat", padx=18, pady=10, cursor="hand2", borderwidth=0,
        )
        b.bind("<Enter>", lambda e: b.config(bg=hover))
        b.bind("<Leave>", lambda e: b.config(bg=color))
        return b

    def process_email(self):
        content = self.email_text.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Empty", "Please paste an email first.")
            return

        features = self.vectorizer.transform([clean(content)])
        label = self.model.predict(features)[0]

        # Probability if available
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            classes = list(self.model.classes_)
            spam_p = float(proba[classes.index("spam")])
        else:
            spam_p = 1.0 if label == "spam" else 0.0

        self._render_result(label, spam_p)

    def _render_result(self, label, spam_p):
        if label == "spam":
            self.result_label.config(text="🚫  SPAM DETECTED", fg=DANGER)
            self.bar_fill.config(bg=DANGER)
        else:
            self.result_label.config(text="✅  Looks legitimate", fg=OK)
            self.bar_fill.config(bg=OK)

        pct = round(spam_p * 100, 1)
        self.confidence_label.config(
            text=f"Spam probability: {pct}%   ·   Ham probability: {round(100 - pct, 1)}%"
        )
        self.bar_bg.update_idletasks()
        full_w = self.bar_bg.winfo_width()
        self.bar_fill.place_configure(relwidth=spam_p)

    def delete_mail(self):
        self.email_text.delete("1.0", tk.END)
        self.result_label.config(text="Awaiting input", fg=TEXT)
        self.confidence_label.config(text="Paste an email and click Check.")
        self.bar_fill.place_configure(relwidth=0)


if __name__ == "__main__":
    root = tk.Tk()
    SpamFilterAIApp(root)
    root.mainloop()
