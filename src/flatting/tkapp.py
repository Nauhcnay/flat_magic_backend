from flatting.app import main as start_server

## Briefcase doesn't support tkinter

import asyncio
import tkinter as tk

async def start_server_in_thread():
    def start_server_wrapper():
        asyncio.set_event_loop(asyncio.new_event_loop())
        start_server()
    
    # Run the server in a thread.
    import threading
    server = threading.Thread( name='flatting_server', target=start_server_wrapper )
    server.setDaemon( True )
    server.start()

def start_gui():
    import tkinter as tk
    root = tk.Tk()
    
    ## Adapting: https://stackoverflow.com/questions/47895765/use-asyncio-and-tkinter-or-another-gui-lib-together-without-freezing-the-gui
    loop = asyncio.get_event_loop()
    
    INTERVAL = 1/30
    async def guiloop():
        while True:
            root.update()
            await asyncio.sleep( INTERVAL )
    task = loop.create_task( guiloop() )
    
    def shutdown():
        task.cancel()
        loop.stop()
        # loop.close()
        root.destroy()
    
    root.title("Flatting Backend")
    ## The port changes if we pass a port argument to `web.run_app`.
    tk.Label( root, text="Serving at http://127.0.0.1:8080" ).pack()
    tk.Button( root, text="Quit", command=shutdown ).pack()
    
    # tk.mainloop()
    

def main():
    start_gui()
    start_server()

if __name__ == '__main__':
    main()
