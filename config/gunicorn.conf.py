import multiprocessing

bind = "0.0.0.0:8000"

workers = multiprocessing.cpu_count() * 1 + 2
worker_connections = 1000
threads = 1
max_requests = 3000

timeout = 600

accesslog = '-'
errorlog = '-'
debug = True
logfile = '/var/log/gunicorn/debug.log'
loglevel = 'debug'
