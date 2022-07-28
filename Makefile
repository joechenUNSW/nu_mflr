CFLAGS=-O3 -fopenmp 
PATHS=-I/usr/local/Cellar/gsl/2.6/include -L/usr/local/Cellar/gsl/2.6/lib/
LIBS=-lgsl -lgslcblas -lm

nu_mflr: nu_mflr.c Makefile pcu.h 
	/usr/local/bin/gcc-11 nu_mflr.c -o nu_mflr $(CFLAGS) $(PATHS) $(LIBS) 

clean:
	$(RM) nu_mflr

