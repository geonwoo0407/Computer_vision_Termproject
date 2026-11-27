"""Publish confirmed camera observations to the SeatSense app."""

import base64
from contextlib import closing
from getpass import getpass
import http.client
import json
import os
import sys
import threading
import time
from urllib.parse import urlsplit

from seat_fusion import MAX_SAMPLE_GAP


class SeatApp:
    def __init__(self, url):
        address = urlsplit(url)
        if (not address.hostname or address.username or address.password or address.query
                or address.fragment or address.path not in ("", "/")):
            raise ValueError("--app-url must be the SeatSense server's base URL.")
        if address.scheme == "https":
            self.connection = http.client.HTTPSConnection
        elif address.scheme == "http" and address.hostname in ("localhost", "127.0.0.1", "::1"):
            self.connection = http.client.HTTPConnection
        else:
            raise ValueError("Use HTTPS for a remote SeatSense server or HTTP on localhost.")
        self.host, self.port = address.hostname, address.port
        password = os.getenv("SEAT_ADMIN_PASSWORD")
        if not password and sys.stdin.isatty():
            password = getpass("SeatSense 앱 관리자 비밀번호: ")
        if not password:
            raise ValueError("Set SEAT_ADMIN_PASSWORD to the app's configured admin password.")
        token = base64.b64encode(f"admin:{password}".encode()).decode()
        self.headers = {"Content-Type": "application/json", "Authorization": "Basic " + token}
        self.delivered = {}
        self.retry_at = 0.0

    def send(self, seat, state, captured):
        now = time.monotonic()
        if self.delivered.get(seat) == state or now < self.retry_at or now-captured > MAX_SAMPLE_GAP:
            return
        payload = dict(action="detect", source="camera", seat_id=seat,
                       occupied=state[0], belongings=state[1])
        try:
            with closing(self.connection(self.host, self.port, timeout=2)) as client:
                client.request("POST", "/api/action", json.dumps(payload), self.headers)
                response = client.getresponse()
                status = response.status
                response.read()
            if status == 403:
                raise ValueError("SeatSense rejected admin authentication. Check SEAT_ADMIN_PASSWORD.")
            if status == 409:
                raise ValueError("In SeatSense /demo select 카메라 API 모드.")
            if 400 <= status < 500:
                raise ValueError(f"SeatSense rejected seat {seat} with HTTP {status}.")
            if status != 200:
                raise OSError(f"SeatSense HTTP {status}")
        except (OSError, http.client.HTTPException) as exc:
            self.delivered.pop(seat, None)
            self.retry_at = time.monotonic() + 2
            print(f"Camera update pending ({exc}); retrying fresh observations.", file=sys.stderr)
            return
        self.delivered[seat] = state
        print(f"SeatSense updated seat {seat}", flush=True)


class GridPublisher:
    """Keep only the latest six observations while the network worker is busy."""

    def __init__(self, app, seat_ids):
        if (not isinstance(seat_ids, (tuple, list)) or len(seat_ids) != 6
                or not all(type(s) is int and 1 <= s <= 288 for s in seat_ids) or len(set(seat_ids)) != 6):
            raise ValueError("--app-seats requires six distinct SeatSense seat numbers (1..288).")
        self.app = app
        self.seat_ids = tuple(seat_ids)
        self.latest = {}
        self.changed = threading.Condition()
        self.pending = False
        self.stopping = False
        self.worker = threading.Thread(target=self._run, name="SeatSense-app")
        self.worker.start()

    def update(self, seats, captured):
        latest = {}
        for seat in seats:
            number = self.seat_ids[seat["slot"] - 1]
            seat["app_seat_id"] = number
            if seat["confirmed_this_frame"]:
                state = seat["state"]
                latest[number] = ((state == "occupied", state == "away"), captured)
        with self.changed:
            self.latest = latest
            self.pending = True
            self.changed.notify()

    def _run(self):
        retry_after = 0.0
        while True:
            with self.changed:
                self.changed.wait_for(lambda: self.pending or self.stopping)
                if not self.pending:
                    return
                self.pending = False
            if time.monotonic() < retry_after:
                continue
            for number in self.seat_ids:
                with self.changed:
                    observation = self.latest.get(number)
                if observation is None:
                    continue
                try:
                    self.app.send(number, *observation)
                except ValueError as exc:
                    print(f"SeatSense: {exc} Retrying in 10s.", file=sys.stderr, flush=True)
                    retry_after = time.monotonic() + 10
                    break

    def close(self):
        with self.changed:
            self.stopping = True
            self.changed.notify()
        self.worker.join()
