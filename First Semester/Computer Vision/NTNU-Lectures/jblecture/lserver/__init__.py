import portpicker
import threading
import http.server
import socketserver
from .jbcd import JBcd

def startLocalServer( ):
    import portpicker
    import threading
    import socket
    from six.moves import socketserver
    from six.moves import SimpleHTTPServer

    class V6Server(socketserver.TCPServer):
        address_family = socket.AF_INET6

    def server_entry():
        if ('HTTPD' in cfg) and ( cfg['HTTPD'] ):
            stopLocalServer()
        cfg['HTTPD'] = None
        handler = SimpleHTTPServer.SimpleHTTPRequestHandler
        port = portpicker.pick_unused_port()
        with JBcd( cfg['REVEAL_DIR'] ):
            os.chdir( cfg['REVEAL_DIR'] )
            httpd = V6Server(("::", port), handler)
            print("serving at port", port, 'cwd', os.getcwd(), 'reveal', cfg['REVEAL_DIR'] )
            cfg['HTTPD'] = httpd
            cfg['HTTP_PORT'] = port
            httpd.serve_forever()


def stopLocalServer():
    httpd = cfg['HTTPD']
    if ( httpd ):
        httpd.shutdown()
        httpd.server_close()
    cfg['HTTPD'] = None
    #thread = cfg['HTTP_LOCALSERVER']
