import asyncio
from concurrent.futures import ProcessPoolExecutor
import functools

import flatting_api

## This controls the number of parallel processes.
## Keep in mind that parallel processes will load duplicate networks
## and compete for the same RAM, which could lead to thrashing.
## Pass `max_workers = N` for exactly `N` parallel processes.
executor = ProcessPoolExecutor()

async def run_async( f ):
    ## We expect this to be called from inside an existing loop.
    ## As a result, we call `get_running_loop()` instead of `get_event_loop()` so that
    ## it raises an error if our assumption is false, rather than creating a new loop.
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor( executor, f )
    return data

async def run_single( *args, **kwargs ):
    return await run_async( functools.partial( flatting_api.run_single, *args, **kwargs ) )

async def merge( *args, **kwargs ):
    return await run_async( functools.partial( flatting_api.merge, *args, **kwargs ) )

async def split_manual( *args, **kwargs ):
    return await run_async( functools.partial( flatting_api.split_manual, *args, **kwargs ) )
