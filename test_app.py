from contextlib import closing
import tempfile, unittest, os
from app import create_app

def authenticate(app, client):
    app.config['STAFF_PASSWORD'] = 'test-only'
    with client.get('/admin', auth=('admin','test-only')) as response:
        assert response.status_code == 200

class SeatTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.app=create_app(os.path.join(self.tmp.name,'test.db')); self.c=self.app.test_client()
        authenticate(self.app,self.c)
    def tearDown(self): self.tmp.cleanup()
    def action(self,action,**kw): return self.c.post('/api/action',json=dict(action=action,seat_id=1,**kw))
    def seat(self): return self.c.get('/api/seats').json['seats'][0]
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
        self.assertEqual(404,self.c.post('/api/action',json={'action':'reserve','seat_id':99}).status_code)
    def test_modes(self):
        self.action('mode',mode='camera');self.assertEqual(409,self.detect(True,True).status_code)
        self.assertEqual(200,self.action('detect',source='camera',occupied=True,belongings=False).status_code)
        self.assertEqual(409,self.action('reset').status_code)
    def test_shared_persistence_and_pages(self):
        self.action('reserve');c2=create_app(self.app.config['DATABASE']).test_client()
        self.assertTrue(c2.get('/api/seats').json['seats'][0]['reserved'])
        for path in ('/','/admin','/demo','/static/app.js','/static/style.css'):
            with self.c.get(path) as response: self.assertEqual(200,response.status_code)
    def test_health_does_not_create_a_session(self):
        client = self.app.test_client()
        response = client.get('/health')
        self.assertEqual({'status': 'ok'}, response.json)
        self.assertNotIn('Set-Cookie', response.headers)
    def test_return_owner_only_and_repeat(self):
        self.action('reserve')
        other=self.app.test_client()
        self.assertFalse(other.get('/api/seats').json['seats'][0]['mine'])
        for act in ('return','cancel'):
            self.assertEqual(403,other.post('/api/action',json={'action':act,'seat_id':1}).status_code)
        self.assertEqual(200,self.action('return').status_code)
        self.assertEqual(409,self.action('return').status_code)
    def test_return_occupied_and_auto_exit(self):
        self.action('reserve');self.detect(True,True)
        self.assertEqual(409,self.action('cancel').status_code)
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
    def test_cancel_distinct_audit(self):
        self.action('reserve');self.action('cancel');self.action('reserve');self.action('return')
        import sqlite3
        with closing(sqlite3.connect(self.app.config['DATABASE'])) as c, c:
            acts=[r[0] for r in c.execute('SELECT action FROM reservation_events ORDER BY id')]
        self.assertEqual(['reserve','cancel','reserve','return'],acts)
    def test_admin_assignment(self):
        other=self.app.test_client();other_id=other.get('/api/seats').json['viewer_id']
        self.action('grant',owner_id=other_id)
        self.assertFalse(self.seat()['mine']);self.assertTrue(other.get('/api/seats').json['seats'][0]['mine'])
        self.assertEqual(200,other.post('/api/action',json={'action':'return','seat_id':1}).status_code)
    def test_migration_preserves_legacy(self):
        import sqlite3
        path=os.path.join(self.tmp.name,'old.db')
        with closing(sqlite3.connect(path)) as c, c:
            c.execute('CREATE TABLE seats (id INTEGER PRIMARY KEY,reserved INTEGER,occupied INTEGER,belongings INTEGER,used INTEGER,away_since REAL,updated REAL)')
            c.execute('INSERT INTO seats VALUES (1,1,0,1,1,1234.5,1235.5)')
        legacy=create_app(path);client=legacy.test_client();authenticate(legacy,client);s=client.get('/api/seats').json['seats'][0]
        self.assertTrue(s['reserved']);self.assertEqual(1234.5,s['away_since']);self.assertIsNone(s['owner_id'])
        self.assertEqual(403,client.post('/api/action',json={'action':'return','seat_id':1}).status_code)
        viewer=client.get('/api/seats').json['viewer_id']
        self.assertEqual(200,client.post('/api/action',json={'action':'assign_owner','seat_id':1,'owner_id':viewer}).status_code)
        self.assertEqual(1234.5,client.get('/api/seats').json['seats'][0]['away_since'])
        create_app(path) # migration is repeatable
    def test_identity_survives_restart(self):
        self.action('reserve');cookie=self.c.get_cookie('session')
        client=create_app(self.app.config['DATABASE']).test_client();client.set_cookie('session',cookie.value)
        self.assertTrue(client.get('/api/seats').json['seats'][0]['mine'])
if __name__=='__main__': unittest.main(verbosity=2)
