
CC = g++

CFLAGS = -std=c++11
 
serial: serial.o load_mnist.o
	$(CC) -o $@ $(CFLAGS) $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f serial *.o
