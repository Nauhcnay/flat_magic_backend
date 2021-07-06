from . import app

if __name__ == '__main__':
    ## https://docs.python.org/3/library/multiprocessing.html#multiprocessing.freeze_support
    if app.MULTIPROCESS: app.multiprocessing.freeze_support()
    app.main()
