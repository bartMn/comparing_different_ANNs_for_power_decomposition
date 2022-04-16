import time 

x = -0.1
start = time.time()
for i in range(10000000):
    x_new = x*x
end = time.time()
t = end - start
print(f"time to ^2: {t}")



start = time.time()
for i in range(10000000):
    x_new = abs(x)
end = time.time()
t = end - start
print(f"time to abs: {t}")