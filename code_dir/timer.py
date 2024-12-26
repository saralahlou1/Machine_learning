import time

class Timer:
    def __init__(self, verbose):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.verbose = verbose

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None  # Reset end time
        self.elapsed_time = None  # Reset elapsed time
        if self.verbose:
            print("Timer started...")

    def stop(self):
        """Stop the timer and calculate the elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        if self.verbose:
            print(f"Timer stopped. Elapsed time: {self.elapsed_time:.4f} seconds")

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        if self.verbose:
            print("Timer reset.")

    def get_elapsed_time(self):
        """Get the elapsed time without stopping the timer."""
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        return time.time() - self.start_time

