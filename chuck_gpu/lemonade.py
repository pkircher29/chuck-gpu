"""Bounded Lemonade cloud session, visible and stoppable in Chuck GPU."""
import argparse
import asyncio
import json
import secrets
import time
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

IMAGE = 'ghcr.io/lemonade-sdk/lemonade-server:v11.9.0'
HOP_HEADERS = {'host', 'connection', 'transfer-encoding', 'content-length',
               'keep-alive', 'upgrade', 'proxy-authorization', 'proxy-authenticate'}
SESSION = web.AppKey('session', aiohttp.ClientSession)


def proxy_app(upstream, api_key):
    app = web.Application(client_max_size=100 * 1024**2)

    async def session_context(app):
        async with aiohttp.ClientSession(auto_decompress=False,
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=30)) as session:
            app[SESSION] = session
            yield
    app.cleanup_ctx.append(session_context)

    async def forward(request):
        if request.host.split(':')[0] not in ('127.0.0.1', 'localhost'):
            raise web.HTTPForbidden()
        origin = request.headers.get('Origin')
        if origin and urlsplit(origin).netloc != request.host:
            raise web.HTTPForbidden()
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in HOP_HEADERS | {'authorization', 'origin'}}
        headers['Authorization'] = 'Bearer ' + api_key
        headers.setdefault('Accept-Encoding', 'identity')
        if request.path.endswith('/health'):
            headers['Accept-Encoding'] = 'identity'
        try:
            async with app[SESSION].request(request.method, upstream + request.raw_path,
                    headers=headers, data=request.content.iter_chunked(65536),
                    allow_redirects=False) as response:
                # Audio websocket tunneling is not implemented. Do not advertise
                # port 9000, which may belong to a different local Lemonade.
                if request.path.endswith('/health') and response.status == 200:
                    data = await response.json()
                    data['websocket_port'] = int(request.host.rsplit(':', 1)[1])
                    return web.json_response(data)
                result = web.StreamResponse(status=response.status,
                    headers={k: v for k, v in response.headers.items()
                             if k.lower() not in HOP_HEADERS})
                await result.prepare(request)
                async for chunk in response.content.iter_any():
                    await result.write(chunk)
                await result.write_eof()
                return result
        except aiohttp.ClientError:
            raise web.HTTPBadGateway(text='Cloud Lemonade is unavailable.') from None
    app.router.add_route('*', '/{path:.*}', forward)
    return app


async def serve(args):
    import modal
    from chuck_gpu.cli import APP_NAME
    from chuck_gpu.jobs import JobRecorder
    root = Path(__file__).resolve().parents[1]
    key = secrets.token_urlsafe(32)
    # Bind locally before allocating a billed GPU, catching port conflicts early.
    runner = web.AppRunner(proxy_app('https://not-ready.invalid', key))
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', args.port)
    await site.start()
    job = JobRecorder(root=root, key='lemonade', command=['lemonade', '--model', args.model],
                      gpu='T4', image=IMAGE, timeout=args.minutes * 60)
    job.update(phase='starting', cpu=2, memory=8192, gpu_source='Lemonade launcher')
    sb = None
    phase, code = 'failed', 1
    try:
        app = await modal.App.lookup.aio(APP_NAME, create_if_missing=True)
        volume = modal.Volume.from_name('chuck-gpu-lemonade-cache', create_if_missing=True)
        image = modal.Image.from_registry(IMAGE).entrypoint([]).dockerfile_commands('USER root')
        config = json.dumps({'llamacpp': {'backend': 'cuda'}, 'ctx_size': 4096,
                             'broadcast': False, 'telemetry': {'enabled': False}})
        script = ('mkdir -p /data/.config/lemonade; '
                  "printf '%s' '" + config + "' > /data/.config/lemonade/config.json; "
                  'exec /opt/lemonade/lemond --host 0.0.0.0 --port 13305')
        print('Starting Lemonade 11.9.0 on T4; maximum session: ' + str(args.minutes) + ' minutes.', flush=True)
        sb = await modal.Sandbox.create.aio('sh', '-c', script,
            app=app, image=image, gpu='T4', cpu=2, memory=8192,
            timeout=args.minutes * 60, volumes={'/data': volume},
            env={'HOME': '/data', 'HF_HOME': '/data/.cache/huggingface'},
            secrets=[modal.Secret.from_dict({'LEMONADE_API_KEY': key, 'LEMONADE_ADMIN_API_KEY': key})],
            encrypted_ports=[13305])
        job.update(sandbox_id=sb.object_id, started=time.time())
        print('Sandbox: ' + sb.object_id, flush=True)
        tunnels = await sb.tunnels.aio()
        url = tunnels[13305].url
        # Replace the placeholder listener only after a tunnel is allocated.
        await runner.cleanup()
        runner = web.AppRunner(proxy_app(url, key))
        await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', args.port).start()
        async with aiohttp.ClientSession(headers={'Authorization': 'Bearer ' + key},
                timeout=aiohttp.ClientTimeout(total=600)) as client:
            for attempt in range(90):
                if await sb.poll.aio() is not None:
                    raise RuntimeError('Cloud server exited: ' + (await sb.stderr.read.aio())[-3000:])
                try:
                    async with client.get(url + '/api/v1/health', timeout=aiohttp.ClientTimeout(total=5)) as r:
                        if r.status == 200:
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(2)
            else:
                raise RuntimeError('Lemonade did not become healthy within three minutes')
            job.update(phase='running', detail='Loading ' + args.model, running_since=time.time())
            print('Server healthy; downloading/loading ' + args.model, flush=True)
            async with client.post(url + '/api/v1/pull', json={'model_name': args.model, 'stream': False}) as r:
                body = await r.text()
                if r.status != 200:
                    raise RuntimeError('Model download failed: ' + body[:1000])
            async with client.post(url + '/api/v1/chat/completions', json={
                    'model': args.model, 'messages': [{'role': 'user', 'content': 'Reply with: Lemonade is ready.'}],
                    'max_tokens': 128, 'stream': False}) as r:
                result = await r.json()
                if r.status != 200 or not result.get('choices'):
                    raise RuntimeError('Inference failed: ' + json.dumps(result)[:1000])
                if not result['choices'][0]['message'].get('content'):
                    raise RuntimeError('Inference produced no visible answer')
                print('Inference verified: ' + json.dumps(result['choices'][0]['message']), flush=True)
            job.update(detail=f'Ready: http://127.0.0.1:{args.port} | {args.model}')
            print(f'READY http://127.0.0.1:{args.port} (Stop GPU in Chuck dashboard to stop)', flush=True)
            while await sb.poll.aio() is None:
                if job.stop_requested():
                    break
                await asyncio.sleep(3)
        phase, code = 'stopped', 0
    except asyncio.CancelledError:
        phase, code = 'stopped', 130
        raise
    except Exception as error:
        job.update(detail=str(error)[:500])
        raise
    finally:
        try:
            if sb is not None:
                await sb.terminate.aio()
                job.update(gpu_released=time.time())
        finally:
            await runner.cleanup()
            job.finish(phase, code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minutes', type=int, choices=range(1, 121), default=30, metavar='1..120')
    parser.add_argument('--port', type=int, default=13306)
    parser.add_argument('--model', default='Qwen3-4B-Instruct-2507-GGUF')
    args = parser.parse_args()
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
