"""
serve.py — Serves the NeuralTrade frontend via Flask.
Run AFTER starting backend/app.py on port 5000.
"""
from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
