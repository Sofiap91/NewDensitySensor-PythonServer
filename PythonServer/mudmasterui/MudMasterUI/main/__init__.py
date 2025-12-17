from flask import Blueprint

# Create a Blueprint named 'main'. Blueprints are a way to organize routes and handlers in Flask applications.
bp = Blueprint('main', __name__)

# Import routes and routes_measure from the MudMasterUI.main module.
# These imports ensure that the routes and route handlers are registered with the Blueprint.
from MudMasterUI.main import routes
from MudMasterUI.main import routes_measure