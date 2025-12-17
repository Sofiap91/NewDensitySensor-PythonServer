from flask import Blueprint

bp = Blueprint('vna', __name__)

from MudMasterUI.vna import routes