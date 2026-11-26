"""Run the existing SeatSense Flask app with a temporary DB and two simulated camera views."""

import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen

from multiview import seat_number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, default=Path.home() / "Desktop/hello_flask")
    parser.add_argument("--flask-python", help="Python with the SeatSense app dependencies")
    parser.add_argument("--seat", type=seat_number, default=1)
    args = parser.parse_args()
    app_dir = args.app.resolve()
    flask_python = args.flask_python or str(app_dir / ".venv-linux/bin/python")
    if not (app_dir / "app.py").is_file():
        parser.error("Point --app at the SeatSense Flask project.")
    password = secrets.token_urlsafe(8)
    with tempfile.TemporaryDirectory(prefix="seatsense-demo-") as tmp:
        env = dict(os.environ, SEAT_DATABASE=str(Path(tmp) / "demo.db"),
                   SEAT_ADMIN_PASSWORD=password, SESSION_COOKIE_SECURE="false")
        # The server binds port zero itself and communicates its assigned port on stdout.
        launcher = (
            "from werkzeug.serving import make_server\n"
            "from app import app\n"
            "server = make_server('127.0.0.1', 0, app, threaded=True)\n"
            "print(server.server_port, flush=True)\n"
            "server.serve_forever()\n"
        )
        with open(Path(tmp) / "server.log", "w+") as log:
            server = subprocess.Popen([flask_python, "-c", launcher], cwd=app_dir, env=env,
                                      stdout=subprocess.PIPE, stderr=log, text=True)
            try:
                port = server.stdout.readline().strip()
                if not port.isdecimal():
                    log.seek(0)
                    raise RuntimeError(f"Could not launch the local SeatSense app. {log.read()}")
                url = f"http://127.0.0.1:{port}"
                auth = base64.b64encode(("admin:" + password).encode()).decode()
                req = Request(url + "/api/action", data=b'{"action":"mode","mode":"camera"}',
                              headers={"Authorization": "Basic " + auth, "Content-Type": "application/json"})
                with urlopen(req, timeout=5) as response:
                    if json.load(response)["mode"] != "camera":
                        raise RuntimeError("Could not enable camera input on the temporary server.")
                print(f"로컬 SeatSense 관리자 화면  {url}/admin", flush=True)
                interactive = sys.stdin.isatty()
                if interactive:
                    print(f"이번 로컬 시연용 로그인  admin / {password}", flush=True)
                    input(f"관리자 화면에서 {args.seat}번 좌석을 확인한 뒤 Enter를 누르세요. ")
                command = [sys.executable, str(Path(__file__).with_name("multiview.py")), "demo",
                           "--seat", str(args.seat), "--app-url", url]
                subprocess.run(command, env=env, check=True)
                if interactive:
                    input("시연 완료. Enter를 누르면 임시 앱이 종료됩니다. ")
            finally:
                server.terminate()
                server.wait(timeout=10)
                server.stdout.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
