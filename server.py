from aiohttp import web
from PIL import Image
import numpy as np
import flatting_api

routes = web.RouteTableDef()

@routes.get('/')
async def hello(request):
    return web.Response(text="Hello, world")

@routes.post('/flatmultiple')
async def flatmultiple( request ):
    data = await request.json()
    
    imgs = [ np.array( Image.fromdata( base64.decode( img ) ) ) for img in data['image'] ]
    nets = data['net']
    radii = data['radius']
    
    flatted = flatting_api.run_multiple( imgs, nets, radii )
    
    result = {}
    result['images'] = [ base64.encode( PIL.fromarray( img ).save( PNG, io.ByteIO ) ) for img in flatted['something'] ]
    
    return web.json_response( result )

## Add more API entry points

app = web.Application()
app.add_routes(routes)
web.run_app(app)

## From JavaScript:
# let result = await fetch( url_of_server.py, { method: 'POST', body: JSON.stringify(data) } ).json();
