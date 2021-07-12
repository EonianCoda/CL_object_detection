DEBUG_FLAG = True

def debug_print(*argv):
    if DEBUG_FLAG:
        for arg in argv:
            print(arg, end=' ')
        print('')