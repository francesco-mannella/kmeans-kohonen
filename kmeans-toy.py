from pylab import *

np.set_printoptions(suppress=True, precision=3)

# dataset 
x =  array([
    [ 1.0, 0.0, 0.0 ],
    [ 0.0, 1.0, 0.0 ],
    [ 0.0, 0.0, 1.0 ] ])
        
# prototipes
c =  array([ 
    [ 0.02, 0.01, 0.03 ],
    [ 0.01, 0.04, 0.07 ],
    [ 0.09, 0.02, 0.02 ] ])

def iter(x, c, eta=0.5):
    # squared distances
    rx = x.reshape(3,1,3) 
    rc = c.reshape(1,3,3)
    n = norm( rx - rc, axis=2)
    wta = eye(3)[argmin(n, 1)]
    dc = eta*wta*(x - c)
    n2 = n**2
    print "squared_distances:" 
    print ("\t   {}"*3).format("c0", "c1", "c2")
    for i in range(3):
        row ="x%d\t"%i
        win ="\t"
        for j in range(3):
            row += "%5.3f\t"%n2[i, j]
            win += "% 5d\t"%wta[i, j]
        print row
        print win

    print "\ncost function:"
    wins = n[arange(3), argmin(n, 1)]
    print ("{:6.4f} + "*2 +"{:6.4f}" + " = {:6.4f}").format(
            *hstack((wins, wins.sum())))
    print "\noptimized weights:"
    print "{}".format(c + dc)
    print
    return dc

for t in range(10):
    print "iter %d ----\n" % t
    c += iter(x, c)


