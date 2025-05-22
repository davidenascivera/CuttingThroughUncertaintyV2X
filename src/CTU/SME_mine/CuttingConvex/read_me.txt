In order to compile and use the C file in python it's necessary to follow few steps:


1. download mingw64 for windows or gcc64 for windows

2. run the command "gcc -fPIC -shared library.c -o libconvexintersect.so -lm"

3. ready to go!