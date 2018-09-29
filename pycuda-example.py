import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import numpy

a = numpy.random.randn(10,100)

#a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)

cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
    __global__ void doublify(float *a)
    {
      int idx = threadIdx.x + threadIdx.y*10;
      a[idx] *= 2;
    }
    """)

start = time.time()
func = mod.get_function("doublify")
func(a_gpu,  block=(100,10,1))
end = time.time()

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
par=end-start
#------------------------------------------------------------------
print 'Parallel took', par, 'seconds.'
print "---"

start = time.time()
b_doubled = numpy.empty_like(a)

for i in range(a.shape[0]):
	for j in range(a.shape[1]):
		b_doubled[i][j]= a[i][j]*2
	

end = time.time()
ser=end-start
print 'Serial took', ser, 'seconds.'
print "---"

print "Times faster:", ser/par
#print "original array:"
#print a
#print "doubled with kernel:"
#print a_doubled


