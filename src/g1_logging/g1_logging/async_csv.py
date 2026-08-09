"""Bounded asynchronous CSV writing for low-impact ROS logging."""

import csv
from pathlib import Path
import queue
import threading


_STOP = object()


class AsyncCsvWriter:
    """Transform queued samples and write them from one background thread."""

    def __init__(
        self,
        path,
        fieldnames,
        row_builder=None,
        max_queue_size=4,
        error_callback=None,
        file_mode='x',
    ):
        if int(max_queue_size) < 1:
            raise ValueError('max_queue_size must be at least 1')
        if file_mode not in ('x', 'w'):
            raise ValueError("file_mode must be 'x' or 'w'")

        self.path = Path(path)
        self.fieldnames = tuple(fieldnames)
        self._row_builder = row_builder or (lambda item: item)
        self._queue = queue.Queue(maxsize=int(max_queue_size))
        self._error_callback = error_callback
        self._file_mode = file_mode
        self._thread = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._submit_lock = threading.Lock()
        self._closing = False
        self._fatal_error = None
        self._dropped_count = 0
        self._written_count = 0
        self._processing_error_count = 0

    @property
    def dropped_count(self):
        return self._dropped_count

    @property
    def written_count(self):
        return self._written_count

    @property
    def processing_error_count(self):
        return self._processing_error_count

    @property
    def error(self):
        return self._fatal_error

    def start(self, timeout_sec=5.0):
        """Start the worker and wait until the CSV header is open."""
        if self._thread is not None:
            raise RuntimeError('AsyncCsvWriter has already been started')
        self._thread = threading.Thread(
            target=self._run,
            name='g1_logging_csv_writer',
            daemon=False,
        )
        self._thread.start()
        if not self._ready.wait(timeout=max(0.0, float(timeout_sec))):
            raise RuntimeError('Timed out while opening the CSV output')
        if self._fatal_error is not None:
            raise RuntimeError(
                f'Could not open CSV output {self.path}: '
                f'{self._fatal_error}'
            ) from self._fatal_error

    def submit(self, item):
        """Queue an item without blocking, dropping the oldest on overflow."""
        with self._submit_lock:
            if (
                self._thread is None
                or self._closing
                or self._fatal_error is not None
            ):
                return False

            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self._dropped_count += 1
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(item)
                    return True
                except queue.Full:
                    self._dropped_count += 1
                    return False

    def close(self, timeout_sec=5.0):
        """Drain accepted samples, close the file, and join the worker."""
        thread = self._thread
        if thread is None:
            return

        with self._submit_lock:
            if not self._closing:
                self._closing = True
                if thread.is_alive():
                    try:
                        self._queue.put(
                            _STOP,
                            timeout=max(0.0, float(timeout_sec)),
                        )
                    except queue.Full as exc:
                        raise RuntimeError(
                            'Timed out while draining the CSV queue'
                        ) from exc

        thread.join(timeout=max(0.0, float(timeout_sec)))
        if thread.is_alive():
            raise RuntimeError('Timed out waiting for the CSV writer to stop')

    def _notify_error(self, exc):
        callback = self._error_callback
        if callback is None:
            return
        try:
            callback(exc)
        except Exception:
            pass

    def _run(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open(
                self._file_mode,
                encoding='utf-8',
                newline='',
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=self.fieldnames,
                    extrasaction='raise',
                )
                writer.writeheader()
                stream.flush()
                self._ready.set()

                while True:
                    item = self._queue.get()
                    try:
                        if item is _STOP:
                            break
                        try:
                            row = self._row_builder(item)
                        except Exception as exc:
                            self._processing_error_count += 1
                            self._notify_error(exc)
                            continue

                        writer.writerow(row)
                        stream.flush()
                        self._written_count += 1
                    finally:
                        self._queue.task_done()
        except Exception as exc:
            self._fatal_error = exc
            self._notify_error(exc)
        finally:
            self._ready.set()
            self._closed.set()
