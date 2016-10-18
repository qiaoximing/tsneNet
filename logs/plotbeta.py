import matplotlib.pyplot as plt

def smooth(l):
    return [(l[i-2]+l[i-1]+l[i]+l[i+1]+l[i+2])/5 for i in range(2,98)]

x=[i for i in range(96)]
y=[1 for i in range(96)]
l0=[]
l002=[]
l005=[]
l01=[]
l02=[]
l05=[]
l1=[]
l2=[]
for i in open("0.log").readlines():
    l0.append(float(i.replace('\n','')))
for i in open("0.02.log").readlines():
    l002.append(float(i.replace('\n','')))
for i in open("0.05.log").readlines():
    l005.append(float(i.replace('\n','')))
for i in open("0.1.log").readlines():
    l01.append(float(i.replace('\n','')))
for i in open("0.2.log").readlines():
    l02.append(float(i.replace('\n','')))
for i in open("0.5.log").readlines():
    l05.append(float(i.replace('\n','')))
for i in open("1.log").readlines():
    l1.append(float(i.replace('\n','')))
for i in open("2.log").readlines():
    l2.append(float(i.replace('\n','')))

l0=smooth(l0)
l002=smooth(l002)
l005=smooth(l005)
l01=smooth(l01)
l02=smooth(l02)
l05=smooth(l05)
l1=smooth(l1)
l2=smooth(l2)

plt.rc('text', usetex=True)
with plt.style.context('classic'):
    plt.plot(x,l0,'black',linewidth=2,label=r'$\beta=0$')
    plt.plot(x,l002,'green',label=r'$\beta=0.02$')
    plt.plot(x,l005,'orange',label=r'$\beta=0.05$')
    plt.plot(x,l01,'red',label=r'$\beta=0.1$')
    plt.plot(x,l02,'blue',label=r'$\beta=0.2$')
    plt.plot(x,l05,'purple',label=r'$\beta=0.5$')
    plt.plot(x,l1,'brown',label=r'$\beta=1$')
    plt.plot(x,l2,'grey',label=r'$\beta=2$')
    plt.plot(x,y,'black',linewidth=2,label='teacher network')
plt.axis([0,96,0.8,4.5])
plt.grid(True)
plt.legend(ncol=2,loc='best')
plt.title(r'different $\beta$ with $\alpha=+\infty$')
plt.xlabel(r'training epoch')
plt.ylabel(r'validation error rate')
plt.show()
