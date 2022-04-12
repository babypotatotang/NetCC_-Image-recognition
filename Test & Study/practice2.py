import threading

def combine_thread(a,b):
    thread = threading.Thread(target=combine,args=(a,b))
    thread.daemon = True
    thread.start()

def combine(a,b):
    x=a
    y=b
    print(x,y)


combine_thread(10,20)
