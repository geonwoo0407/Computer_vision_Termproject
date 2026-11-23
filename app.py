import hashlib
import os, sqlite3, time, secrets, json
from threading import Condition
from contextlib import closing
from datetime import timedelta
from flask import Flask, g, render_template, request, jsonify, session, Response, send_from_directory
from werkzeug.security import check_password_hash, generate_password_hash

SEAT_COUNT = 288

def create_app(database=None):
    app = Flask(__name__)
    app.jinja_env.globals['seat_count'] = SEAT_COUNT
    app.config['DATABASE'] = (database or os.environ.get('SEAT_DATABASE')
                              or os.path.join(app.instance_path, 'seats.db'))
    os.makedirs(os.path.dirname(os.path.abspath(app.config['DATABASE'])), exist_ok=True)
    app.secret_key = os.environ.get('FLASK_SECRET_KEY')
    if not app.secret_key:
        key_path = app.config['DATABASE'] + '.secret'
        try:
            with open(key_path, 'x') as f: f.write(secrets.token_hex(32))
        except FileExistsError:
            pass
        with open(key_path) as f: app.secret_key = f.read()
    app.config.update(PERMANENT_SESSION_LIFETIME=timedelta(days=365), SESSION_REFRESH_EACH_REQUEST=False,
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax',
                      SESSION_COOKIE_SECURE=os.environ.get('SESSION_COOKIE_SECURE', '').lower()
                      in ('1', 'true', 'yes', 'on'))
    def connect():
        c = sqlite3.connect(app.config['DATABASE'], timeout=10)
        c.row_factory = sqlite3.Row
        return c
    with closing(connect()) as c, c:
        c.execute('CREATE TABLE IF NOT EXISTS seats (id INTEGER PRIMARY KEY, reserved INTEGER DEFAULT 0, occupied INTEGER DEFAULT 0, belongings INTEGER DEFAULT 0, used INTEGER DEFAULT 0, away_since REAL, updated REAL)')
        c.executemany('INSERT OR IGNORE INTO seats(id) VALUES (?)', ((i,) for i in range(1, SEAT_COUNT + 1)))
        c.execute('CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, mode TEXT)')
        c.execute("INSERT OR IGNORE INTO settings VALUES (1,'demo')")
        # Additive migration: preserve every existing row, reservation and timestamp.
        columns = {r[1] for r in c.execute('PRAGMA table_info(seats)')}
        for name in ('owner_id', 'belongings_owner_id'):
            if name not in columns: c.execute(f'ALTER TABLE seats ADD COLUMN {name} TEXT')
        c.execute('CREATE TABLE IF NOT EXISTS visitors (id TEXT PRIMARY KEY)')
        c.execute("""CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY, login_id TEXT UNIQUE COLLATE NOCASE NOT NULL,
            name TEXT NOT NULL, password_hash TEXT NOT NULL)""")
        c.execute('CREATE TABLE IF NOT EXISTS reservation_events (id INTEGER PRIMARY KEY, seat_id INTEGER, action TEXT, actor TEXT, owner_id TEXT, created REAL)')
        c.execute('CREATE TABLE IF NOT EXISTS changes (id INTEGER PRIMARY KEY, revision INTEGER NOT NULL)')
        c.execute('INSERT OR IGNORE INTO changes VALUES (1, 0)')
        c.execute("""CREATE TABLE IF NOT EXISTS reservations (
            id INTEGER PRIMARY KEY, seat_id INTEGER, owner_id TEXT, status TEXT,
            created REAL, modified REAL)""")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS active_seat ON reservations(seat_id) WHERE status='active'")
        c.execute("""INSERT INTO reservations(seat_id,owner_id,status,created,modified)
                     SELECT id,owner_id,'active',NULL,NULL FROM seats s WHERE reserved=1
                     AND NOT EXISTS (SELECT 1 FROM reservations WHERE seat_id=s.id AND status='active')""")
        c.execute("""CREATE TABLE IF NOT EXISTS receipts (viewer TEXT, token TEXT, payload TEXT,
                     code INTEGER, error TEXT, PRIMARY KEY(viewer,token))""")
    app.config['STAFF_PASSWORD'] = os.environ.get('SEAT_ADMIN_PASSWORD')

    def staff_tag():
        password = app.config['STAFF_PASSWORD']
        secret = hashlib.sha256(app.secret_key.encode()).digest()
        return hashlib.blake2b(password.encode(), key=secret, digest_size=24).hexdigest() if password else None

    def staff():
        return bool(staff_tag() and session.get('staff') == staff_tag())

    def staff_page(template):
        if not staff():
            auth = request.authorization
            password = app.config['STAFF_PASSWORD']
            if not password:
                return '관리자 비밀번호 SEAT_ADMIN_PASSWORD 설정이 필요합니다.', 503
            if not auth or auth.username != 'admin' or not secrets.compare_digest((auth.password or '').encode(), password.encode()):
                return Response(status=401, headers={'WWW-Authenticate':'Basic realm="Seat staff"'})
            session['staff'] = staff_tag()
        return render_template(template)
    changed = Condition()

    def available(seat):
        return not (seat['reserved'] or seat['occupied'] or seat['belongings'] or seat['away_since'] is not None)

    @app.before_request
    def identify():
        if not request.path.startswith('/api/'): return
        g.student = None
        if 'student_id' in session:
            with closing(connect()) as c:
                g.student = c.execute('SELECT id,name FROM students WHERE id=?', (session['student_id'],)).fetchone()
            if g.student is None:
                session.pop('student_id')
        if g.student is None and 'viewer_id' not in session:
            session['viewer_id'] = secrets.token_hex(8)
            session.permanent = True
        g.viewer = g.student['id'] if g.student else session['viewer_id']
        expected = request.headers.get('X-Seat-Viewer')
        if expected is not None and expected != g.viewer:
            return jsonify(error='로그인 계정이 변경되었습니다. 좌석을 다시 확인하세요.', identity_changed=True), 409

    @app.get('/login')
    @app.get('/register')
    def account_page():
        return render_template('account.html', registering=request.path == '/register')

    @app.post('/api/student/<operation>')
    def student_account(operation):
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error='JSON 객체가 필요합니다.'), 400
        if operation == 'logout':
            session.clear()
            return jsonify(ok=True)
        if operation not in ('register', 'login'):
            return jsonify(error='지원하지 않는 계정 요청입니다.'), 404
        login_id, password = data.get('login_id'), data.get('password')
        if not isinstance(login_id, str) or not isinstance(password, str):
            return jsonify(error='아이디와 비밀번호를 입력하세요.'), 400
        login_id = login_id.strip()
        if not 1 <= len(login_id) <= 64 or not 1 <= len(password) <= 128:
            return jsonify(error='아이디 또는 비밀번호 길이를 확인하세요.'), 400
        with closing(connect()) as c, c:
            if operation == 'register':
                name = data.get('name')
                if not isinstance(name, str) or not 1 <= len(name.strip()) <= 40:
                    return jsonify(error='이름을 1~40자로 입력하세요.'), 400
                if len(password) < 8:
                    return jsonify(error='비밀번호를 8자 이상 입력하세요.'), 400
                student_id = 'student-' + secrets.token_hex(16)
                inserted = c.execute("""INSERT INTO students(id, login_id, name, password_hash)
                    VALUES (?,?,?,?) ON CONFLICT(login_id) DO NOTHING""",
                    (student_id, login_id, name.strip(), generate_password_hash(password)))
                if not inserted.rowcount:
                    return jsonify(error='이미 사용 중인 아이디입니다.'), 409
            else:
                row = c.execute('SELECT id,password_hash FROM students WHERE login_id=?', (login_id,)).fetchone()
                if not row or not check_password_hash(row['password_hash'], password):
                    return jsonify(error='아이디 또는 비밀번호를 확인하세요.'), 401
                student_id = row['id']
        session.clear()
        session['student_id'] = student_id
        session.permanent = True
        return jsonify(ok=True)

    def view_context():
        return dict(viewer=g.viewer, viewer_name=g.student['name'] if g.student else None,
                    privileged=request.args.get('view') == 'admin' and staff())

    def snapshot(c, viewer, viewer_name, privileged):
        now = time.time()
        seats = []
        query = ('SELECT s.*, a.name AS owner_name FROM seats s LEFT JOIN students a ON s.owner_id=a.id ORDER BY s.id'
                 if privileged else 'SELECT * FROM seats ORDER BY id')
        for row in c.execute(query):
            s = dict(row)
            for k in ('reserved','occupied','belongings','used'): s[k] = bool(s[k])
            s['mine'] = viewer_name is not None and s['reserved'] and s['owner_id'] == viewer
            s['away'] = s['away_since'] is not None
            s['color'] = 'reserved' if s['reserved'] else 'unreserved' if s['occupied'] or s['away'] or s['belongings'] else 'empty'
            s['warning'] = s['color'] == 'unreserved'
            s['status'] = '자리 비움' if s['away'] else '이용 중' if s['occupied'] else '짐만 감지 · 확인 필요' if s['belongings'] else '입실 대기' if s['reserved'] else '빈 좌석'
            if not privileged:
                s.pop('owner_id')
                s.pop('belongings_owner_id')
            seats.append(s)
        return dict(revision=c.execute('SELECT revision FROM changes').fetchone()[0], viewer_id=viewer,
                    viewer_name=viewer_name, seats=seats, mode=c.execute('SELECT mode FROM settings').fetchone()[0], server_time=now)

    @app.after_request
    def private_api(response):
        if request.path.startswith('/api/'):
            response.cache_control.no_store = True
        return response
    @app.get('/sw.js')
    def worker():
        return send_from_directory(app.static_folder, 'sw.js', max_age=0)

    @app.get('/api/live')
    def live():
        context = view_context()
        def events():
            last = -1
            while True:
                with changed:
                    with closing(connect()) as db:
                        db.execute('BEGIN')
                        revision = db.execute('SELECT revision FROM changes').fetchone()[0]
                        packet = snapshot(db, **context) if revision != last else None
                    if packet is None:
                        changed.wait(timeout=5)
                if packet is None:
                    yield 'event: heartbeat\ndata: {}\n\n'
                else:
                    last = revision
                    yield 'data: ' + json.dumps(packet) + '\n\n'
        return Response(events(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    @app.get('/manifest.webmanifest')
    def manifest():
        from flask import send_from_directory
        response = send_from_directory(app.root_path, 'manifest.webmanifest', mimetype='application/manifest+json')
        response.headers['Cache-Control'] = 'no-cache'
        return response
    @app.get('/health')
    def health():
        with closing(connect()) as c:
            c.execute('SELECT 1').fetchone()
        return jsonify(status='ok')
    @app.get('/')
    def student(): return render_template('student.html')
    @app.get('/admin')
    def admin(): return staff_page('admin.html')
    @app.get('/demo')
    def demo(): return staff_page('demo.html')
    @app.get('/api/seats')
    def seats():
        with closing(connect()) as c, c:
            c.execute('BEGIN')
            return jsonify(snapshot(c, **view_context()))
    @app.post('/api/action')
    def action():
        data = request.get_json(silent=True)
        if not isinstance(data, dict): return jsonify(error='JSON 객체가 필요합니다.'),400
        act = data.get('action')
        if act in ('reserve','move','cancel','return') and g.student is None:
            return jsonify(error='학생 로그인 후 이용하세요.'), 401
        with closing(connect()) as c, c:
            c.execute('BEGIN IMMEDIATE')
            token = data.get('request_id')
            viewer = g.viewer
            payload = json.dumps(data, sort_keys=True)
            if act in ('grant','assign_owner','revoke','mode','reset','detect') and not staff():
                auth = request.authorization
                password = app.config['STAFF_PASSWORD']
                if not (password and auth and auth.username == 'admin' and secrets.compare_digest((auth.password or '').encode(), password.encode())):
                    return jsonify(error='관리자 인증이 필요합니다.'),403
            if token is not None:
                if not isinstance(token,str) or not 1 <= len(token) <= 100:
                    return jsonify(error='잘못된 요청 ID'),400
                old = c.execute('SELECT * FROM receipts WHERE viewer=? AND token=?',(viewer,token)).fetchone()
                if old:
                    if old['payload'] != payload:
                        return jsonify(error='요청 ID가 이미 사용되었습니다.'),409
                    if old['error']:
                        return jsonify(error=old['error']),old['code']
                    return jsonify(snapshot(c, **view_context()))
            failure = mutate(c, data, act)
            code, error = (failure[1], failure[0].get_json()['error']) if failure else (200,None)
            if token:
                c.execute('INSERT INTO receipts VALUES (?,?,?,?,?)',(viewer,token,payload,code,error))
            if failure:
                return failure
            result = snapshot(c, **view_context())
        with changed:
            changed.notify_all()
        return jsonify(result)

    @app.post('/api/result')
    def result():
        data = request.get_json(silent=True)
        token = data.get('request_id') if isinstance(data,dict) else None
        if not isinstance(token,str) or not 1 <= len(token) <= 100:
            return jsonify(error='잘못된 요청 ID'),400
        with closing(connect()) as c, c:
            c.execute('BEGIN IMMEDIATE')
            # Close an unreceived request so a delayed packet cannot apply later.
            c.execute('INSERT OR IGNORE INTO receipts VALUES (?,?,NULL,409,?)',
                      (g.viewer,token,'서버에 접수되지 않은 요청입니다.'))
            row = c.execute('SELECT code,error FROM receipts WHERE viewer=? AND token=?',
                            (g.viewer,token)).fetchone()
            return jsonify(code=row['code'],error=row['error'],state=snapshot(c, **view_context()))

    def mutate(c, data, act):
        mode = c.execute('SELECT mode FROM settings').fetchone()[0]
        if act == 'mode':
            if data.get('mode') not in ('demo','camera'): return jsonify(error='잘못된 모드'),400
            c.execute('UPDATE settings SET mode=?', (data['mode'],))
        elif act == 'reset':
            if mode != 'demo': return jsonify(error='시연 모드에서만 초기화 가능합니다.'),409
            c.execute('UPDATE seats SET reserved=0,occupied=0,belongings=0,used=0,away_since=NULL,updated=NULL,owner_id=NULL,belongings_owner_id=NULL')
        else:
            if type(data.get('seat_id')) is not int: return jsonify(error='seat_id는 정수여야 합니다.'),400
            row = c.execute('SELECT * FROM seats WHERE id=?',(data['seat_id'],)).fetchone()
            if row is None: return jsonify(error='존재하지 않는 좌석'),404
            s = dict(row)
            previous_owner = s['owner_id']
            if act in ('reserve','grant','assign_owner'):
                if act == 'reserve' and not available(s):
                    return jsonify(error='현재 예약할 수 없는 좌석입니다.'),409
                if act == 'assign_owner' and not s['reserved']:
                    return jsonify(error='예약된 좌석만 소유자를 지정할 수 있습니다.'),409
                owner = g.student['id'] if g.student else None
                if act != 'reserve' and ('login_id' in data or 'owner_id' in data or act == 'assign_owner'):
                    if 'login_id' in data:
                        key, value = 'login_id', data['login_id']
                    else:
                        key, value = 'id', data.get('owner_id')
                    account = (c.execute(f'SELECT id FROM students WHERE {key}=?', (value.strip(),)).fetchone()
                               if isinstance(value, str) else None)
                    if account is None:
                        return jsonify(error='가입한 학생의 로그인 아이디를 확인하세요.'),400
                    owner = account['id']
                if act == 'grant' and s['reserved']:
                    return jsonify(error='이미 예약된 좌석입니다.'),409
                if c.execute('SELECT 1 FROM seats WHERE reserved=1 AND owner_id=? AND id<>?', (owner, s['id'])).fetchone():
                    return jsonify(error='이미 예약한 좌석이 있습니다. 자리 이동을 이용하세요.'),409
                s.update(reserved=1, owner_id=owner)
            elif act == 'move':
                if not s['reserved'] or previous_owner != g.viewer:
                    return jsonify(error='본인 예약 좌석에서만 이동할 수 있습니다.'),403
                target_id = data.get('target_id')
                if type(target_id) is not int or target_id == s['id']:
                    return jsonify(error='다른 좌석을 선택하세요.'),400
                target = c.execute('SELECT * FROM seats WHERE id=?', (target_id,)).fetchone()
                if target is None or not available(target):
                    return jsonify(error='이동할 자리를 사용할 수 없습니다. 기존 예약은 유지됩니다.'),409
                c.execute('UPDATE seats SET reserved=1, owner_id=? WHERE id=?', (previous_owner,target_id))
                c.execute('INSERT INTO reservation_events(seat_id,action,actor,owner_id,created) VALUES (?,?,?,?,?)',
                          (target_id,'move_in',previous_owner,previous_owner,time.time()))
                s.update(reserved=0, owner_id=None)
            elif act in ('cancel','return','revoke'):
                if act != 'revoke':
                    if not s['reserved']: return jsonify(error='이미 해제된 예약입니다.'),409
                    if s['owner_id'] != g.viewer:
                        return jsonify(error='본인이 예약한 좌석만 취소하거나 반납할 수 있습니다.'),403
                    if act == 'cancel' and s['used']:
                        return jsonify(error='이용을 시작한 좌석은 반납하기를 이용하세요.'),409
                s.update(reserved=0, owner_id=None)
            elif act == 'detect':
                source = data.get('source')
                if source != mode: return jsonify(error='현재 모드와 입력 출처가 다릅니다.'),409
                if any(type(data.get(k)) is not bool for k in ('occupied','belongings')):
                    return jsonify(error='occupied, belongings는 true/false여야 합니다.'),400
                if 'belongings_owner_id' in data:
                    owner = data['belongings_owner_id']
                    if owner is not None and (not isinstance(owner, str) or len(owner) > 128):
                        return jsonify(error='소지품 소유자 ID는 문자열 또는 null이어야 합니다.'),400
                    s['belongings_owner_id'] = owner
                if not data['belongings']: s['belongings_owner_id'] = None
                s.update(occupied=data['occupied'], belongings=data['belongings'], updated=time.time())
                if s['occupied']: s.update(used=1, away_since=None)
                elif s['used'] and (s['reserved'] or s['belongings']):
                    if s['away_since'] is None: s['away_since'] = time.time()
                else: s.update(used=0, away_since=None)
            else: return jsonify(error='알 수 없는 동작'),400
            if not s['reserved'] and not s['occupied'] and not s['belongings']: s.update(used=0,away_since=None)
            c.execute('UPDATE seats SET reserved=?,occupied=?,belongings=?,used=?,away_since=?,updated=?,owner_id=?,belongings_owner_id=? WHERE id=?', tuple(s[k] for k in ('reserved','occupied','belongings','used','away_since','updated','owner_id','belongings_owner_id','id')))
            if act in ('reserve','grant','assign_owner','cancel','return','revoke','move'):
                c.execute('INSERT INTO reservation_events(seat_id,action,actor,owner_id,created) VALUES (?,?,?,?,?)',
                          (s['id'],act,g.viewer,s['owner_id'] or previous_owner,time.time()))
        now = time.time()
        if act in ('reserve','grant'):
            c.execute("INSERT INTO reservations(seat_id,owner_id,status,created,modified) VALUES (?,?,'active',?,?)",
                      (s['id'],s['owner_id'],now,now))
        elif act == 'reset':
            c.execute("UPDATE reservations SET status='reset',modified=? WHERE status='active'", (now,))
        elif act in ('move','assign_owner','cancel','return','revoke'):
            c.execute("""UPDATE reservations SET seat_id=?,owner_id=?,status=?,modified=?
                         WHERE seat_id=? AND status='active'""",
                      (target_id if act=='move' else s['id'], s['owner_id'] or previous_owner,
                       'active' if act in ('move','assign_owner') else act,now,s['id']))
        c.execute('UPDATE changes SET revision=revision+1 WHERE id=1')
    return app

app = create_app()
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')),
            debug=False, threaded=True)
