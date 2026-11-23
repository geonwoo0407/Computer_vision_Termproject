import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from app import create_app
from test_app import register

def authenticate(app, client):
    app.config['STAFF_PASSWORD'] = 'test-only'
    with client.get('/admin', auth=('admin','test-only')) as response:
        assert response.status_code == 200



class SharedRoom(TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        app = create_app(str(Path(tmp.name) / 'seats.db'))
        self.a, self.b = app.test_client(), app.test_client()
        for index, client in enumerate((self.a, self.b)):
            register(client, 'student' + str(index))
            self.assertEqual(len(client.get('/api/seats').json['seats']), 287)

    def simultaneous(self, first, second):
        gate = threading.Barrier(2)
        def submit(client, data):
            gate.wait(timeout=3)
            return client.post('/api/action', json=data).status_code
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(submit, client, data) for client, data
                       in zip((self.a, self.b), (first, second))]
            return [f.result() for f in futures]

    def test_reserve_race(self):
        req = dict(action='reserve', seat_id=12)
        self.assertCountEqual(self.simultaneous(req, req), [200, 409])

    def test_move_race_preserves_loser(self):
        for client, seat in ((self.a, 1), (self.b, 2)):
            client.post('/api/action', json=dict(action='reserve', seat_id=seat))
        codes = self.simultaneous(dict(action='move', seat_id=1,target_id=12),
                                  dict(action='move', seat_id=2,target_id=12))
        self.assertCountEqual(codes, [200,409])
        for client, source, code in zip((self.a,self.b),(1,2),codes):
            seats = client.get('/api/seats').json['seats']
            self.assertEqual([s['id'] for s in seats if s['mine']], [12 if code==200 else source])

    def test_stream_and_move_keep_tracking(self):
        authenticate(self.a.application,self.a)
        authenticate(self.b.application,self.b)
        stream = self.b.get('/api/live?view=admin', buffered=False)
        self.addCleanup(stream.close)
        messages = iter(stream.response)
        next(messages)
        for data in [dict(action='reserve'),dict(action='detect', source='demo', occupied=True, belongings=True),
                     dict(action='detect', source='demo', occupied=False, belongings=True, belongings_owner_id='bag-A')]:
            res = self.a.post('/api/action', json=dict(seat_id=1, **data))
        since = res.json['seats'][0]['away_since']
        req = dict(action='move',seat_id=1,target_id=12)
        self.assertEqual(self.b.post('/api/action',json=req).status_code, 403)
        self.assertEqual(self.a.post('/api/action',json=req).status_code, 200)
        packet = json.loads(next(messages).decode().removeprefix('data: '))
        old,new = packet['seats'][0],packet['seats'][11]
        self.assertEqual((old['away_since'],old['belongings_owner_id'],old['color']),
                         (since,'bag-A','unreserved'))
        self.assertTrue(new['reserved'])
        self.assertFalse(new['mine'] or new['occupied'])

    def test_missing_response_and_delayed_packet(self):
        for index, action in enumerate(('reserve','move','return','reserve','cancel')):
            seat = 2 if action=='return' else 1
            body = dict(action=action,seat_id=seat,request_id=str(index))
            if action=='move':
                body['target_id']=2
            self.a.post('/api/action', json=body)  # Discard the response.
            result = self.a.post('/api/result',json={'request_id':str(index)}).json
            self.assertEqual(result['code'],200)
            self.assertEqual(self.a.post('/api/action',json=body).json['revision'],index+1)
        self.assertEqual(self.a.post('/api/result',json={'request_id':'late'}).json['code'],409)
        self.assertEqual(self.a.post('/api/action',json=dict(action='reserve',seat_id=1,request_id='late')).status_code,409)

    def test_reconnect_after_another_user_reserves(self):
        self.b.post('/api/action',json=dict(action='reserve',seat_id=3))
        stream=self.a.get('/api/live',buffered=False)
        with stream:
            seat=json.loads(next(iter(stream.response)).decode().removeprefix('data: '))['seats'][2]
            self.assertTrue(seat['reserved'])
            self.assertFalse(seat['mine'])
            self.assertNotIn('owner_id',seat)
        self.assertEqual(self.a.post('/api/action',json=dict(action='revoke',seat_id=3)).status_code,403)

    def test_many_clients(self):
        import time
        clients=[self.a.application.test_client() for _ in range(24)]
        for index, client in enumerate(clients):
            register(client, 'racer' + str(index))
        gate=threading.Barrier(len(clients))
        def compete(pair):
            i,client=pair
            client.get('/api/seats')
            gate.wait(timeout=15)
            start=time.perf_counter()
            response=client.post('/api/action',json=dict(action='reserve',seat_id=1+i%6,request_id=str(i)))
            return response.status_code, (time.perf_counter()-start)*1000
        with ThreadPoolExecutor(len(clients)) as pool:
            results=list(pool.map(compete,enumerate(clients)))
        codes,times=zip(*results)
        self.assertCountEqual(codes,[200]*6+[409]*18)
        print(f'24 clients: 6 accepted, 18 conflicts, mean {sum(times)/24:.1f} ms, max {max(times):.1f} ms')
