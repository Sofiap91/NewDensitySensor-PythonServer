from enum import Enum


class TeltonikaEndpoints(Enum):
    """Enumeration of Teltonika router API endpoints."""
    ip_address = '192.168.1.1'
    username = 'admin'
    password = 'MudM45t3r'

class VNAConfig(Enum):
    """Enumeration of VNA configuration parameters."""
    port = 5000