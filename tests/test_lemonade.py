import unittest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestServer
from chuck_gpu.lemonade import proxy_app


class ProxyTests(AioHTTPTestCase):
    async def get_application(self):
        async def upstream(request):
            if request.headers.get('Authorization') != 'Bearer test-key':
                return web.Response(status=401)
            if request.path == '/stream':
                response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
                await response.prepare(request)
                await response.write(b'data: one\n\n')
                await response.write(b'data: [DONE]\n\n')
                return response
            if request.path == '/missing':
                return web.Response(status=404, text='missing model')
            if request.path == '/api/v1/health':
                self.assertEqual(request.headers.get('Accept-Encoding'), 'identity')
                return web.json_response({'status': 'ok', 'websocket_port': 9000})
            return web.json_response({'path': request.path, 'body': await request.text()})
        app = web.Application()
        app.router.add_route('*', '/{path:.*}', upstream)
        self.upstream = TestServer(app)
        await self.upstream.start_server()
        return proxy_app(str(self.upstream.make_url('')).rstrip('/'), 'test-key')

    async def tearDownAsync(self):
        await super().tearDownAsync()
        await self.upstream.close()

    async def test_forwards_api_with_authentication(self):
        response = await self.client.post('/api/v1/chat/completions', json={'model': 'small'})
        self.assertEqual(response.status, 200)
        data = await response.json()
        self.assertEqual(data['path'], '/api/v1/chat/completions')
        self.assertIn('small', data['body'])

    async def test_rejects_foreign_browser_origin(self):
        response = await self.client.post('/api/v1/load', headers={'Origin': 'https://evil.example'})
        self.assertEqual(response.status, 403)

    async def test_rejects_foreign_host(self):
        response = await self.client.get('/', headers={'Host': 'evil.example'})
        self.assertEqual(response.status, 403)

    async def test_streams_sse_and_preserves_errors(self):
        response = await self.client.get('/stream')
        self.assertEqual(response.headers['Content-Type'], 'text/event-stream')
        self.assertEqual(await response.text(), 'data: one\n\ndata: [DONE]\n\n')
        response = await self.client.get('/missing')
        self.assertEqual(response.status, 404)
        self.assertEqual(await response.text(), 'missing model')

    async def test_health_does_not_route_voice_to_other_local_server(self):
        response = await self.client.get('/api/v1/health', headers={'Accept-Encoding': 'gzip'})
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())['websocket_port'], self.server.port)


if __name__ == '__main__':
    unittest.main()
