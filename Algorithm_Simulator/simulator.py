import tkinter as tk
from tkinter import ttk
import sys
import os

class SimulatorStartPage(tk.Tk):
    def __init__(self):
        super().__init__()


        self.title("Algorithm Simulator")
        self.geometry("700x500")
        self.configure(bg="#f0f2f5")
        self.resizable(False, False)

        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#f0f2f5")
        self.style.configure("TLabel", background="#f0f2f5", font=("Segoe UI", 11))
        self.style.configure("Title.TLabel", font=("Segoe UI", 26, "bold"), foreground="#2c3e50")
        self.style.configure("Subtitle.TLabel", font=("Segoe UI", 13), foreground="#555")
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=10)
        
        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Algorithm Simulator", style="Title.TLabel")
        title_label.pack(pady=(20, 10))

        subtitle_label = ttk.Label(main_frame, text="Choose your simulation mode", style="Subtitle.TLabel")
        subtitle_label.pack(pady=(0, 30))

        # Cards Frame
        cards_frame = ttk.Frame(main_frame)
        cards_frame.pack(padx=30, pady=10, fill=tk.X)

        # Maze Card
        self.create_card(cards_frame,
                        title="Maze Simulator",
                        desc="Explore mazes using pathfinding algorithms.",
                        button_text="Launch Maze",
                        command=self.launch_maze_simulator).grid(row=0, column=0, padx=20)

        # Puzzle Card
        self.create_card(cards_frame,
                        title="Puzzle Simulator",
                        desc="Solve puzzles using smart search algorithms.",
                        button_text="Launch Puzzle",
                        command=self.launch_puzzle_simulator).grid(row=0, column=1, padx=20)

        # Footer
        footer = ttk.Frame(main_frame)
        footer.pack(pady=(40, 10))

        ttk.Button(footer, text="Exit", command=self.destroy).pack()

    def create_card(self, parent, title, desc, button_text, command):
        card = ttk.Frame(parent, padding=20, relief="raised", borderwidth=2)
        
        title_label = ttk.Label(card, text=title, font=("Segoe UI", 14, "bold"))
        title_label.pack(pady=(0, 10))

        desc_label = ttk.Label(card, text=desc, wraplength=200, justify="center")
        desc_label.pack(pady=(0, 20))

        button = ttk.Button(card, text=button_text, command=command)
        button.pack()

        return card

    def launch_maze_simulator(self):
        self.destroy()
        from MazeSimulator import MazeSimulator
        app = MazeSimulator()
        app.mainloop()

    def launch_puzzle_simulator(self):
        self.destroy()
        from puzzle_simulator import PuzzleSimulator
        app = PuzzleSimulator()
        app.mainloop()

if __name__ == "__main__":
    app = SimulatorStartPage()
    app.mainloop()
