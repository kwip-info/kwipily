"""Bound cooperative work between live authority checks, never cache authority.

Scalar row loops share a row/cell/text-work boundary. Entry, completion, I/O,
publication and SQLite interruption guards remain the caller's responsibility.
Unknown structured cells conservatively force a check for that row.
"""


class RowCheckpoints:
    def __init__(self, check):
        self.check = check
        self.rows = self.cells = self.text = 0
        self.started = False

    def admit(self, row):
        text = sum(4 * len(value) if type(value) is str else
                   65_536 if type(value) not in (int, float, bool, type(None)) else 0
                   for value in row)
        if (not self.started or self.rows >= 128 or
                self.cells + len(row) > 4096 or self.text + text > 65_536):
            self.check()
            self.rows = self.cells = self.text = 0
            self.started = True
        self.rows += 1
        self.cells += len(row)
        self.text += text

    def finish(self):
        self.check()
