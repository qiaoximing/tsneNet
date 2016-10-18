import matplotlib.pyplot as plt

def smooth(l):
    return [(l[i-2]+l[i-1]+l[i]+l[i+1]+l[i+2])/5 for i in range(2,98)]

x=[i for i in range(96)]
y=[1 for i in range(96)]
l0=[]
l002=[]
l005=[]
l01=[]
for i in open("0.log").readlines():
    l0.append(float(i.replace('\n','')))
for i in open("0.01-5d.log").readlines():
    l002.append(float(i.replace('\n','')))
for i in open("0.05-10d.log").readlines():
    l005.append(float(i.replace('\n','')))
for i in open("0.1.log").readlines():
    l01.append(float(i.replace('\n','')))

l0=smooth(l0)
l002=smooth(l002)
l005=smooth(l005)
l01=smooth(l01)

plt.rc('text', usetex=True)
with plt.style.context('classic'):
    plt.plot(x,l0,'black',linewidth=2,label=r'$\beta=0$')
    plt.plot(x,l002,'green',label=r'$\alpha=5,~\beta=0.01$')
    plt.plot(x,l005,'orange',label=r'$\alpha=10,~\beta=0.05$')
    plt.plot(x,l01,'red',label=r'$\alpha=+\infty,~\beta=0.1$')
    plt.plot(x,y,'black',linewidth=2,label='teacher network')
plt.axis([0,96,0.8,4.5])
plt.grid(True)
plt.legend(ncol=2,loc='best')
plt.title(r'different $\alpha$ with optimal $\beta$')
plt.xlabel(r'training epoch')
plt.ylabel(r'validation error rate')
plt.show()
