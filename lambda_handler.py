import json
import base64
from mangum import Mangum
from app import app

# Create Mangum handler
handler = Mangum(app)