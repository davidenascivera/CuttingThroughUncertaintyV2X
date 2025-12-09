# CuttingThroughUncertainty – C Library Usage

This project includes a C library (`convex_intersect_lib.c`) that can be compiled and used in Python. Below are the instructions for both **Windows** and **Ubuntu** environments.

---

## 🪟 Windows Instructions

To compile and use the C file in Python on Windows:

1. **Install a C Compiler**  
   Download and install [MinGW-w64](https://www.mingw-w64.org/) or a similar GCC-based compiler for Windows.

2. **Compile the C File as a Shared Library**  
   Open PowerShell or Git Bash, navigate to the folder containing `convex_intersect_lib.c`, and run:
   ```bash
   gcc -shared -o libconvexintersect.dll convex_intersect_lib.c -lm
   ```

3. ✅ **Done!**  
   You can now load `libconvexintersect.dll` in Python using `ctypes` or `cffi`.

---

## 🐧 Ubuntu Instructions

To build the same shared library on Ubuntu:

1. **Install GCC (if not already installed)**
   ```bash
   sudo apt update
   sudo apt install build-essential
   ```

2. **Compile the C File**
   ```bash
   gcc -fPIC -shared library.c -o libconvexintersect.so -lm
   ```

3. ✅ **Ready to use in Python**

---

## 🐍 Example: Load the Shared Library in Python

Here's how to load and use the compiled `.so` file in Python:

```python
import ctypes

# Load the shared library
lib = ctypes.CDLL('./libconvexintersect.so')

# Suppose the library has a function: int add(int, int)
lib.add.argtypes = [ctypes.c_int, ctypes.c_int]
lib.add.restype = ctypes.c_int

# Call the function
result = lib.add(3, 5)
print("Result of add(3, 5):", result)
```

> ⚠️ Replace `add` with the actual function(s) defined in `library.c`.

---

## 📁 File Structure Example

```
FinalProject/
├── library.c
├── libconvexintersect.so
├── your_python_script.py
└── README.md
```

---
