import sys

if __name__ == '__main__':
    sys.stdout.write('output\n')
    sys.stdout.flush()
    sys.stderr.write('error\n')
    sys.stderr.flush()
    if len(sys.argv) > 1:
        sys.exit(1)
