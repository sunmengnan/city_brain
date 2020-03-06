from eve import Eve

app = Eve(settings='link/config.py')
app.run(host="0.0.0.0", port=5001)
