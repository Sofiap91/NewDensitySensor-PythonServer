from flask import Blueprint

# Create a new Blueprint named 'eventStreams'
bp = Blueprint('eventStreams', __name__)

# Import routes associated with the 'eventStreams' Blueprint
# This should be done after creating the Blueprint to avoid circular imports
from MudMasterUI.eventStreams import routes