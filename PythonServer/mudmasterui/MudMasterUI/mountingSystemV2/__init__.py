"""
*******************************************************************************
@file   MudMasterUI.mountingSystemV2.__init__.py
@author Scott Thomason
@date   16 Aug 2024
@brief  Initialization file for the mountingSystemV2 module. Sets up the 
        Flask Blueprint for mountingSystemV2 and imports the routes.

REFERENCE:

*******************************************************************************
"""

""" Imports
*******************************************************************************
"""
from flask import Blueprint  # Import the Blueprint class from Flask

# Create a Blueprint named 'mountingSystemV2' for the mountingSystemV2 module
bp = Blueprint('mountingSystemV2', __name__)

# Import routes for the mountingSystemV2 module to register them with the Blueprint
from MudMasterUI.mountingSystemV2 import routes