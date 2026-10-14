"""Networkless container: local HTTP relay can only call its message broker."""
import http.server
import json
import os
import subprocess
import sys
import threading
from .isolation_wire import call

class Relay(http.server.BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*args):pass
    def do_POST(self):
        if self.path.split('?')[0] != '/v1/messages':
            self.send_error(404);return
        try:
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            result=call('/broker/messages.sock',dict(body=payload,headers={k:v for k,v in self.headers.items() if k.lower().startswith('anthropic-')}))
            body=result['body'].encode()
            self.send_response(result['status']);self.send_header('Content-Type',result['content_type'])
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        except Exception as exc:
            self.send_error(502,str(exc))

def main():
    relay=http.server.ThreadingHTTPServer(('127.0.0.1',0),Relay)
    thread=threading.Thread(target=relay.serve_forever,daemon=True);thread.start()
    env={**os.environ,'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{relay.server_port}',
         'ANTHROPIC_API_KEY':'local-broker-not-a-provider-key',
         'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1',
         'CLAUDE_CONFIG_DIR':'/tmp/claude-config'}
    try:code=subprocess.call(sys.argv[1:],env=env)
    finally:relay.shutdown()
    raise SystemExit(code)

if __name__=='__main__':main()
