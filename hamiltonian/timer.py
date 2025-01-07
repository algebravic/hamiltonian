from time import process_time
class Timer:

    def __init__(self, name: str):
        self._name = name
    def __enter__(self):
        self._start = process_time()
    def __exit__(self, *args):
        end = process_time()
        print(f"{self._name}: {end - self._start} seconds")
