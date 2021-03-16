from aiohttp import web
from PIL import Image
from io import BytesIO

import numpy as np
import flatting_api
import base64
import io

routes = web.RouteTableDef()

# initial models
assert flatting_api.initial_nets()

@routes.get('/')
# seems the function name is not that important?
async def hello(request):
    return web.Response(text="Flatting API server is running")

@routes.post('/flatmultiple')
async def flatmultiple( request ):
    # read input
    # how to test my API if I want to send a post request?
    # I guess data is python object already
    # this may similar to json.loads()
    data = await request.json()
    
    # convert data to function input
    # img should be a PIL.Image
    # net should be string
    # radius should be int
    # preview should be boolean
    imgs = [ to_pil(byte) for byte in data['image'] ]
    nets = data['net']
    radii = data['radius']
    preview = data['preview']

    # run function in back end
    flatted = flatting_api.run_multiple(imgs, nets, radii, preview)
    
    # construct and return the result
    result = {}
    result['image'] = [ to_base64(img) for img in flatted['fill_color'] ]
    result['fillmap'] = [fillmap.tolist() for fillmap in flatted['fill_integer'] ]
    # layers would be more complex, but it should bascially same as image
    # may we should let layers generated at client side, that will reduce a lot of computation and transmission time
    result['layers'] = []
    for layers in flatted['layers']:
        result['layers'].append([ to_base64(img) for img in layers ])
    
    result['image_c'] = [ to_base64(img) for img in flatted['components_color'] ]
    result['fillmap_c'] = [fillmap_c.tolist()  for fillmap_c in flatted['components_integer'] ]
    result['layers_c'] = []
    for layers_c in flatted['components_layers']:
        result['layers_c'] = [ to_base64(img) for img in layers_c ]

    result['palette'] = [palette.tolist() for palette in flatted['palette']]

    # https://docs.aiohttp.org/en/stable/web_reference.html
    # Return Response with predefined 'application/json' content type and data encoded by dumps parameter (json.dumps() by default).
    return web.json_response( result )

## Add more API entry points
@routes.post('/flatsingle')
async def flatsingle( request ):
    data = await request.json()
    img = to_pil(data['image'])
    net = data['net']
    radii = data['radius']
    preview = data['preview']

    flatted = flatting_api.run_single(img, net, radii, preview)

    result = {}
    result['image'] = to_base64(flatted['fill_color'])
    result['fillmap'] = flatted['fill_integer'].tolist()
    # layers would be more complex, but it should bascially same as image
    result['layers'] = [to_base64(img) for img in flatted['layers']]
    
    result['image_c'] = to_base64(flatted['components_color'])
    result['fillmap_c'] = flatted['components_integer'].tolist()
    result['layers_c'] = [to_base64(img) for img in flatted['components_layers']]
    result['palette'] = flatted['palette'].tolist()

    return web.json_response( result )
                
@routes.post('/refreshnet')
async def refreshnet( request ):
    return web.json_response(flatting_api.initial_nets(True))

@routes.post('/merge')
async def merge( request ):
    data = await request.json()
    
    img = np.array(to_pil(data['image']))
    stroke = np.array(to_pil(data['stroke']))
    palette = np.array(data['palette'])
    
    merged = flatting_api.merge(img, stroke, palette)

    result = {}
    result['image'] = to_base64(merged['fill_color'])
    result['fillmap'] = merged['fill_integer'].tolist()
    result['layers'] = [to_base64(img) for img in merged['layers']]
    result['palette'] = merged['palette'].tolist()

    return web.json_response(result)

@routes.post('/splitauto')
async def split_auto( request ):
    data = await request.json()
    
    img = to_pil(data['image'])
    img_artist = to_pil(data['image_artist'])
    stroke = to_pil(data['stroke'])
    palette = np.array(data['palette'])
    
    splited = flatting_api.split_auto(img, img_artist, stroke, palette)

    result = {}
    result['image'] = to_base64(splited['fill_color'])
    result['fillmap'] = splited['fill_integer'].tolist()
    result['layers'] = [to_base64(img) for img in splited['layers']]
    result['palette'] = splited['palette'].tolist()

    return web.json_response(result)


@routes.post('/splitmanual')
async def split_manual( request ):
    data = await request.json()
    
    img = to_pil(data['image'])
    img_artist = to_pil(data['image_artist'])
    stroke = to_pil(data['stroke'])
    palette = np.array(data['palette'])
    
    splited = flatting_api.split_manual(img, img_artist, stroke, palette)

    result = {}
    result['image'] = to_base64(splited['fill_color'])
    result['fillmap'] = splited['fill_integer'].tolist()
    result['layers'] = [to_base64(img) for img in splited['layers']]
    result['palette'] = splited['palette'].tolist()

    return web.json_response(result)    

@routes.post('/showfillmap')
async def show_fillmap_manual( request ):
    data = await request.json()

    img = to_pil(data['image'])
    palette = np.array(data['palette'])

    fill_color = flatting_api.show_fillmap_manual(img, palette)

    result = {}
    result["image"] = to_base64(fill_color["fill_color"])
    result["palette"] = fill_color["palette"].tolist()

    return web.json_response(result)

def to_base64(array):
    '''
    A helper function to convert numpy array to png in base64 format
    '''
    with io.BytesIO() as output:
        Image.fromarray(array).save(output, format='png')
        img = output.getvalue() 
    return img

def to_pil(byte):
    '''
    A helper function to convert byte png to PIL.Image
    '''
    byte = base64.decodebytes(byte)
    return Image.open(BytesIO(byte))

app = web.Application()
app.add_routes(routes)
web.run_app(app)

## From JavaScript:
# let result = await fetch( url_of_server.py, { method: 'POST', body: JSON.stringify(data) } ).json();
