import itertools, math
import time


def func(x, y, z):
    #part1 = 2*(y+1)**5 + 3*(x-3)**3 + 4*x - 15*math.sin(3.14/3*x) + 20*math.sin(z)
    #part2 = 5*(x+2)**2 - 3*(y+4) - 4*y**2 -z
    #part3 = 16*math.sin(z/5)+  6*math.sin(3.14/10*y) + 13*math.sin(x*10)
    
    #part1 = math.sin(x*2) + math.sin(y/3) + math.sin(z*10) 
    #part2 = math.sin(x/2) + math.sin(y/8) + math.sin(z/10) 
    #part3 = math.sin(x*6) + math.sin(y*5) + math.sin(z*3/4) 
    #return (part1**2 + part2**3 + part3**4)
    x = 1.0*x/100
    y = 1.0*y/100
    z = 1.0*z/100
    return math.sin(3*x + y**2 + 1.5*z)

def bad_min(X, Y, Z):

    start = time.time()
    
    variables = [X, Y, Z]
    possibillities = list(itertools.product(*variables))
    min_value = 99999999999999999999999
    x_min = None
    y_min = None
    z_min = None
    
    for x,y,z in possibillities:
        temp = func(x, y, z)
        if min_value > temp:
            min_value = temp
            x_min = x
            y_min = y
            z_min = z
    
    end = time.time()
    
    print(f"min value for all possibilities is {min_value}")
    print(f"x is: {x_min}")
    print(f"y is: {y_min}")
    print(f"z is: {z_min}")
    print(f"time: {end - start}")
    print()
    
def test_min(X, Y, Z):
    t= int(len(X)/2)
    x_min = X[0]
    y_min = Y[0]
    z_min = Z[0]
    start = time.time()
    min_value = 9999999999999999999999999
    for i in range(10):
        
        for y in Y:
            temp = func(x_min, y, z_min)
            if min_value > temp:
                min_value = temp
                y_min = y
            
        for z in Z:
            temp = func(x_min, y_min, z)
            if min_value > temp:
                min_value = temp
                z_min = z
                
        for x in X:
            temp = func(x, y_min, z_min)
            if min_value > temp:
                min_value = temp
                x_min = x
                
    end = time.time()
    
    print(f"min value for test method is {min_value}")
    print(f"x is: {x_min}")
    print(f"y is: {y_min}")
    print(f"z is: {z_min}")
    print(f"time: {end - start}")
    print()
            
    
#x = range(-100, 100)
#y = range(-100, 100)
#z = range(-100, 100)

x = range(0, 314)
y = range(0, 314)
z = range(0, 314)


bad_min(x, y, z)
test_min(x, y, z)

