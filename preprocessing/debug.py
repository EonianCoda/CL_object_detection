DEBUG_FLAG = True

def debug_print(**kwargs):
    if DEBUG_FLAG:
        print(**kwargs)