"""
Main Application Entry Point
Initializes Flask server and main controller for VNA data collection
"""
from flask import Flask, render_template, jsonify, request
import constants as const
from teltonika_interface import TeltonikaInterface
from vna_interface import VNAInterface
from main_controller import MainController

# Initialize Flask app
app = Flask(__name__)

# Initialize main controller
controller = MainController(vna_server_url="http://127.0.0.1:5000")


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html', title='Home')


@app.route('/api/start-measurements', methods=['POST'])
def start_measurements():
    """Start measurement process."""
    result = controller.start_measurements()
    return jsonify(result)


@app.route('/api/stop-measurements', methods=['POST'])
def stop_measurements():
    """Stop measurement process."""
    result = controller.stop_measurements()
    return jsonify(result)


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status."""
    status = controller.get_status()
    return jsonify(status)


if __name__ == '__main__':
    print("Starting VNA Data Collection System on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=True)
