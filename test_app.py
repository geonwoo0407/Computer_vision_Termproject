from contextlib import closing
import re
import tempfile, unittest, os
from app import create_app

def register(client, number, name='테스트 학생'):
    response = client.post('/api/student/register', json=dict(student_number=number, name=name, password='seat-test123'))
    assert response.status_code == 200, response.json

def authenticate(app, client):
    app.config['STAFF_PASSWORD'] = 'test-only'
    with client.post('/api/login', json={'student_number':'admin', 'password':'test-only'}) as response:
        assert response.status_code == 200

class SeatTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.app=create_app(os.path.join(self.tmp.name,'test.db')); self.c=self.app.test_client()
        register(self.c, '00240001', '김하늘')
        self.manager = self.app.test_client()
        authenticate(self.app,self.manager)
    def tearDown(self): self.tmp.cleanup()
    def action(self,action,**kw):
        client = self.manager if action in ('detect','grant','revoke','mode','reset','assign_owner') else self.c
        return client.post('/api/action',json=dict(action=action,seat_id=1,**kw))
    def seat(self): return self.manager.get('/api/seats?view=admin').json['seats'][0]
    def detect(self,p,b): return self.action('detect',source='demo',occupied=p,belongings=b)
    def test_unreserved_lifecycle(self):
        self.detect(True,True);self.assertTrue(self.seat()['warning'])
        self.detect(False,True);t=self.seat()['away_since'];self.assertIsNotNone(t)
        self.detect(False,True);self.assertEqual(t,self.seat()['away_since'])
        self.detect(True,True);self.assertFalse(self.seat()['away'])
        self.detect(False,False);self.assertEqual('empty',self.seat()['color'])
    def test_reserved_lifecycle(self):
        self.action('reserve');self.assertEqual('입실 대기',self.seat()['status'])
        self.detect(True,False);self.detect(False,False)
        self.assertTrue(self.seat()['away']);self.assertEqual('reserved',self.seat()['color'])
        self.action('return');self.assertEqual('empty',self.seat()['color'])
    def test_grant_revoke_preserves_timer(self):
        self.detect(True,True);self.detect(False,True);t=self.seat()['away_since']
        self.action('grant');self.assertEqual('reserved',self.seat()['color'])
        self.action('revoke');self.assertEqual(t,self.seat()['away_since']);self.assertTrue(self.seat()['warning'])
    def test_unknown_bag(self):
        self.detect(False,True);self.assertFalse(self.seat()['away']);self.assertEqual('짐만 감지 · 확인 필요',self.seat()['status'])
    def test_conflict_and_validation(self):
        self.action('reserve');self.assertEqual(409,self.action('reserve').status_code)
        self.assertEqual(400,self.action('detect',source='demo',occupied='false',belongings=True).status_code)
        self.assertEqual(404,self.c.post('/api/action',json={'action':'reserve','seat_id':289}).status_code)
    def test_modes(self):
        self.action('mode',mode='camera');self.assertEqual(409,self.detect(True,True).status_code)
        self.assertEqual(200,self.action('detect',source='camera',occupied=True,belongings=False).status_code)
        self.assertEqual(409,self.action('reset').status_code)
    def test_shared_persistence_and_pages(self):
        self.action('reserve');c2=create_app(self.app.config['DATABASE']).test_client()
        self.assertTrue(c2.get('/api/seats').json['seats'][0]['reserved'])
        for path in ('/','/admin','/demo','/static/app.js','/static/style.css'):
            with self.manager.get(path) as response: self.assertEqual(200,response.status_code)
    def test_floor_plan_numbers_match_the_reading_room(self):
        for path in ('/', '/admin', '/demo'):
            page = self.manager.get(path).get_data(as_text=True)
            ids = [int(value) for value in re.findall(r'data-seat-id="(\d+)"', page)]
            self.assertEqual(sorted(ids), list(range(1, 289)))
            self.assertEqual(ids[:14], [12, 13, 11, 14, 10, 15, 9, 16, 8, 17, 7, 18, 6, 19])
            self.assertEqual(ids[22:26], [1, 24, 36, 37])
            self.assertEqual(ids[142:146], [121, 144, 156, 157])
            self.assertEqual(ids[-6:], [267, 286, 266, 287, 265, 288])
        self.assertEqual(self.c.post('/api/action', json={'action':'reserve','seat_id':288}).status_code,200)
        seats = self.c.get('/api/seats').json['seats']
        self.assertTrue(seats[-1]['mine'])

    def test_health_does_not_create_a_session(self):
        client = self.app.test_client()
        response = client.get('/health')
        self.assertEqual({'status': 'ok'}, response.json)
        self.assertNotIn('Set-Cookie', response.headers)
    def test_return_owner_only_and_repeat(self):
        self.action('reserve')
        other=self.app.test_client()
        register(other, '00240002')
        self.assertFalse(other.get('/api/seats').json['seats'][0]['mine'])
        self.assertEqual(403,other.post('/api/action',json={'action':'return','seat_id':1}).status_code)
        self.assertEqual(200,self.action('return').status_code)
        self.assertEqual(409,self.action('return').status_code)
    def test_return_occupied_and_auto_exit(self):
        self.action('reserve');self.detect(True,True)
        self.action('return');self.assertEqual('unreserved',self.seat()['color'])
        self.assertTrue(self.seat()['occupied']);self.assertTrue(self.seat()['used'])
        self.detect(False,True);self.assertTrue(self.seat()['away'])
        self.detect(False,False);self.assertEqual('empty',self.seat()['color'])
    def test_return_bags_preserves_attribution(self):
        self.action('reserve');self.detect(True,True)
        self.action('detect',source='demo',occupied=False,belongings=True,belongings_owner_id='person-track-7')
        t=self.seat()['away_since'];self.action('return')
        self.assertEqual('unreserved',self.seat()['color'])
        self.assertEqual(t,self.seat()['away_since'])
        self.assertEqual('person-track-7',self.seat()['belongings_owner_id'])
        self.detect(False,False);self.assertIsNone(self.seat()['belongings_owner_id'])
    def test_return_unknown_bags_no_false_away(self):
        self.action('reserve');self.detect(False,True);self.action('return')
        self.assertEqual('unreserved',self.seat()['color']);self.assertFalse(self.seat()['away'])
    def test_return_releases_a_new_reservation(self):
        self.action('reserve')
        self.assertEqual(400, self.action('cancel').status_code)
        self.assertTrue(self.seat()['reserved'])
        self.assertEqual(200, self.action('return').status_code)
        import sqlite3
        with closing(sqlite3.connect(self.app.config['DATABASE'])) as c, c:
            acts=[r[0] for r in c.execute('SELECT action FROM reservation_events ORDER BY id')]
        self.assertEqual(['reserve','return'],acts)
    def test_admin_assignment(self):
        other=self.app.test_client();register(other,'00240002','이민서')
        self.action('grant',student_number='00240002')
        self.assertEqual(self.seat()['owner_name'], '이민서')
        self.assertFalse(self.seat()['mine']);self.assertTrue(other.get('/api/seats').json['seats'][0]['mine'])
        self.assertEqual(200,other.post('/api/action',json={'action':'return','seat_id':1}).status_code)
    def test_migration_preserves_legacy(self):
        import sqlite3
        path=os.path.join(self.tmp.name,'old.db')
        with closing(sqlite3.connect(path)) as c, c:
            c.execute('CREATE TABLE seats (id INTEGER PRIMARY KEY,reserved INTEGER,occupied INTEGER,belongings INTEGER,used INTEGER,away_since REAL,updated REAL)')
            c.execute('INSERT INTO seats VALUES (1,1,0,1,1,1234.5,1235.5)')
        legacy=create_app(path);client=legacy.test_client();register(client,'00240003');manager=legacy.test_client();authenticate(legacy,manager);s=manager.get('/api/seats?view=admin').json['seats'][0]
        self.assertTrue(s['reserved']);self.assertEqual(1234.5,s['away_since']);self.assertIsNone(s['owner_id'])
        self.assertEqual(403,client.post('/api/action',json={'action':'return','seat_id':1}).status_code)
        viewer=client.get('/api/seats').json['viewer_id']
        self.assertEqual(200,manager.post('/api/action',json={'action':'assign_owner','seat_id':1,'owner_id':viewer}).status_code)
        self.assertEqual(1234.5,client.get('/api/seats').json['seats'][0]['away_since'])
        create_app(path) # migration is repeatable
    def test_identity_survives_restart(self):
        self.action('reserve');cookie=self.c.get_cookie('session')
        client=create_app(self.app.config['DATABASE']).test_client();client.set_cookie('session',cookie.value)
        self.assertTrue(client.get('/api/seats').json['seats'][0]['mine'])

    def test_account_owns_reservation_across_devices_and_logout(self):
        guest = self.app.test_client()
        with guest.session_transaction() as sess:
            sess['viewer_id'] = 'stale-browser-id'
        reserve = {'action':'reserve','seat_id':1,'owner_id':'untrusted'}
        self.assertEqual(guest.post('/api/action',json=reserve).status_code,401)
        self.assertEqual(self.c.post('/api/action',json=reserve).status_code,200)
        login = {'student_number':'00240001','password':'seat-test123'}
        self.assertEqual(guest.post('/api/student/login',json={**login,'password':'wrong'}).status_code,401)
        self.assertEqual(guest.post('/api/student/register',json={**login,'name':'다른 이름'}).status_code,409)
        self.assertEqual(guest.post('/api/student/login',json=login).status_code,200)
        self.assertEqual(guest.get('/api/seats').json['viewer_name'],'김하늘')
        self.assertTrue(guest.get('/api/seats').json['seats'][0]['mine'])
        self.assertEqual(guest.post('/api/action',json={**reserve,'seat_id':2}).status_code,409)
        self.assertEqual(guest.post('/api/student/logout',json={}).status_code,200)
        self.assertIsNone(guest.get('/api/seats').json['viewer_name'])
        self.assertEqual(guest.post('/api/action',json={'action':'return','seat_id':1}).status_code,401)
        self.assertTrue(self.seat()['reserved'])

    def test_names_only_on_authenticated_admin_responses(self):
        self.action('reserve')
        self.detect(True,False)
        self.assertEqual(self.seat()['owner_name'],'김하늘')
        other = self.app.test_client()
        register(other,'00240002','이민서')
        for client, query in [(self.c,''), (other,''), (other,'?view=admin')]:
            def redacted(seat):
                self.assertNotIn('owner_name',seat)
                self.assertNotIn('owner_id',seat)
            redacted(client.get('/api/seats'+query).json['seats'][0])
            with client.get('/api/live'+query,buffered=False) as stream:
                import json
                packet = json.loads(next(iter(stream.response)).decode().removeprefix('data: '))
                redacted(packet['seats'][0])
            redacted(client.post('/api/result'+query,json={'request_id':'missing'}).json['state']['seats'][0])
        self.assertEqual(other.post('/api/action',json={'action':'return','seat_id':1}).status_code,403)

    def test_personal_ticket_favorites_and_stale_return(self):
        other = self.app.test_client()
        register(other, '00240002')
        first = self.action('reserve').json['my_reservation']
        self.assertGreater(first['created'], 0)
        self.assertEqual(first['seat_id'], 1)
        self.assertEqual(self.action('favorite').json['favorites'], [1])
        other_state = other.get('/api/seats').json
        self.assertIsNone(other_state['my_reservation'])
        self.assertEqual(other_state['favorites'], [])
        moved = self.action('move', reservation_id=first['id'], target_id=138).json['my_reservation']
        self.assertEqual(moved, {**first, 'seat_id':138})
        self.assertIsNone(self.c.post('/api/action', json={'action':'return','seat_id':138,
                          'reservation_id':first['id']}).json['my_reservation'])
        second = self.action('reserve').json['my_reservation']
        self.assertNotEqual(second['id'], first['id'])
        self.assertEqual(self.action('return', reservation_id=first['id']).status_code, 409)
        self.assertEqual(self.c.get('/api/seats').json['my_reservation'], second)
        self.assertEqual(self.action('unfavorite').json['favorites'], [])

    def test_unified_login_and_student_directory(self):
        other = self.app.test_client()
        register(other, '00240002', '김하늘')
        guest = self.app.test_client()
        for client in (guest, other):
            for path in ('/admin', '/demo', '/admin/students'):
                self.assertEqual(client.get(path).location, '/login')
            self.assertEqual(client.post('/admin/students', data={'q':'김하늘'}).location, '/login')
        self.assertEqual(guest.post('/api/login', json={'student_number':'admin','password':'wrong'}).status_code,401)
        for invalid in ('abc', 'admin', '１２３'):
            response = guest.post('/api/student/register', json={'student_number':invalid, 'name':'학생', 'password':'seat-test123'})
            self.assertEqual(response.status_code, 400)
        response = self.manager.post('/admin/students', data={'q':'김하늘'})
        self.assertTrue(response.cache_control.no_store)
        self.assertEqual(response.text.count('<td>김하늘</td>'), 2)
        for query, expected in [('00240002', 1), ('0024', 0), ('하늘', 2)]:
            response = self.manager.post('/admin/students', data={'q':query})
            self.assertEqual(response.text.count('<td>김하늘</td>'), expected)
        self.assertEqual(self.manager.post('/api/login', json={'student_number':'00240001','password':'seat-test123'}).json['role'], 'student')
        self.assertEqual(self.manager.get('/admin/students').location, '/login')
        self.assertEqual(other.post('/api/login', json={'student_number':'admin','password':'test-only'}).json['role'], 'admin')
        self.assertIsNone(other.get('/api/seats').json['viewer_name'])
        other.post('/api/student/logout', json={})
        self.assertEqual(other.get('/admin/students').location, '/login')

    def test_login_has_only_registration_and_no_linking_route(self):
        page = self.c.get('/login')
        self.assertIn('href="/register"', page.text)
        self.assertNotIn('link-student-number', page.text)
        self.assertEqual(self.c.get('/link-student-number').status_code, 404)
        self.assertEqual(self.c.post('/api/student/link-number', json={}).status_code, 404)

    def test_account_changes_reject_stale_tab_action(self):
        before = self.c.get('/api/seats').json['viewer_id']
        register(self.c,'00240002')
        response = self.c.post('/api/action',json={'action':'reserve','seat_id':1},headers={'X-Seat-Viewer':before})
        self.assertTrue(response.json['identity_changed'])
        self.assertFalse(self.seat()['reserved'])
if __name__=='__main__': unittest.main(verbosity=2)
