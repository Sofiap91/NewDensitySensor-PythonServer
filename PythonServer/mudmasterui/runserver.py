"""
This script runs the MudMasterUI application using a development server.
"""

from MudMasterUI import createApp

if __name__ == '__main__':
    
    app = createApp()
    app.run('0.0.0.0', 80, threaded=True)
    #app.run('192.168.1.102', 80, threaded=True)
