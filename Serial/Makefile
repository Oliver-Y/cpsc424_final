
CC = g++

CFLAGS = -g -std=c++11 
 
serial: serial.o mnist.o
	$(CC) -o $@ $(CFLAGS) $^

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f serial *.o