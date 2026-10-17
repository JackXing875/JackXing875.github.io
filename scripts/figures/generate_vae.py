#!/usr/bin/env python3
"""Reproducible vector schematics for the VAE/DDPM tutorial (no measured data)."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-blog-vae')
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch, Circle
N='#243746'; T='#267D8D'; O='#C77C46'; G='#84939C'; L='#DCE3E6'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':N,'axes.labelcolor':N,'mathtext.fontset':'stix','svg.fonttype':'path','svg.hashsalt':'vae-paper-figures-v1','pdf.fonttype':42,'savefig.facecolor':'white','lines.linewidth':1.5})
parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',default=str(Path(__file__).parent)); P=Path(parser.parse_args().output_dir); P.mkdir(parents=True,exist_ok=True)
def base(h=4.2):
 f,a=plt.subplots(figsize=(10,h)); a.set(xlim=(0,10),ylim=(0,h)); a.axis('off'); f.subplots_adjust(left=.025,right=.975,bottom=.035,top=.965); return f,a
def tx(a,x,y,s,size=11,c=N,ha='center',**kw): a.text(x,y,s,fontsize=size,color=c,ha=ha,va='center',**kw)
def ar(a,x1,y1,x2,y2,c=G,style='-',rad=0):
 a.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,linewidth=1.3,color=c,linestyle=style,connectionstyle=f'arc3,rad={rad}'))
def mod(a,x,y,w,h,label,c=T,enc=True):
 pts=[(x-w/2,y-h/2),(x-w/2,y+h/2),(x+w/2,y+h*.28),(x+w/2,y-h*.28)]
 if not enc: pts=[(x-w/2,y-h*.28),(x-w/2,y+h*.28),(x+w/2,y+h/2),(x+w/2,y-h/2)]
 a.add_patch(Polygon(pts,fc='white',ec=c,lw=1.7)); tx(a,x,y,label,13,c)
def save(f,name):
 for ext in ['svg','pdf','png']: f.savefig(P/(name+'.'+ext),dpi=180,metadata=({'Creator':'Python / Matplotlib','Date':None} if ext=='svg' else {'Creator':'Python / Matplotlib','CreationDate':None,'ModDate':None} if ext=='pdf' else None))
 plt.close(f)
def density(a,x,y,w=1.0,h=.55,c=T):
 u=np.linspace(-3,3,100); v=np.exp(-u*u/2); a.plot(x+u*w/6,y+v*h,c=c,lw=1.7); a.plot([x-w*.55,x+w*.55],[y,y],c=L,lw=.8)
# 1: two explicit computational paths with reusable decoder.
f,a=base(4.4)
tx(a,.2,4.05,'a',12,ha='left',fontweight='bold'); tx(a,.55,4.05,'Training',11,ha='left')
tx(a,.55,2.8,'$x$',23); mod(a,2,2.8,1.0,1.25,r'$q_\phi$',T)
tx(a,2,1.93,'Encoder',10,T); ar(a,.85,2.8,1.4,2.8)
density(a,3.6,2.46,.95,.65,T); tx(a,3.6,2.12,r'$\mu_\phi,\;\sigma_\phi$',13); ar(a,2.6,2.8,2.98,2.8)
a.add_patch(Circle((5.25,2.8),.27,fc='white',ec=O,lw=1.5)); tx(a,5.25,2.8,'$z$',18,O); ar(a,4.22,2.8,4.92,2.8)
tx(a,5.25,3.85,r'$\epsilon\sim\mathcal{N}(0,I)$',14,O); ar(a,5.25,3.56,5.25,3.12,O)
tx(a,5.25,2.05,r'$z=\mu_\phi+\sigma_\phi\odot\epsilon$',14,O)
mod(a,7,2.8,1.0,1.25,r'$p_\theta$',T,False); ar(a,5.57,2.8,6.4,2.8); tx(a,7,1.93,'Decoder',10,T)
ar(a,7.6,2.8,8.4,2.8); tx(a,9,2.8,r'$p_\theta(x\mid z)$',17); tx(a,9,2.12,'Likelihood',10)
a.plot([.2,9.8],[1.57,1.57],c=L,lw=.8)
tx(a,.2,1.16,'b',12,ha='left',fontweight='bold'); tx(a,.55,1.16,'Generation',11,ha='left')
tx(a,3,.59,r'$z\sim\mathcal{N}(0,I)$',16,O); ar(a,4.15,.59,6.4,.59,O)
mod(a,7,.59,1,.9,r'$p_\theta$',T,False); ar(a,7.6,.59,8.5,.59); tx(a,8.98,.59,r'$x\sim p_\theta(\cdot\mid z)$',15)
save(f,'01-vae-paths')
# 2: vector point distributions; forward noise progressively erases a data manifold.
f,a=base(3.65)
xs=[1.1,3.7,6.3,8.9]; ab=[1,.7,.24,.002]; labs=['$x_0$','$x_{t-1}$','$x_t$','$x_T$']
for x,b,label in zip(xs,ab,labs):
 # Illustrative marginal states sharing a Gaussian draw; not an empirical trajectory.
 # Individual vector cells; clipping is for the display palette only.
 grid=np.linspace(-1,1,16); X,Y=np.meshgrid(grid,grid); signal=np.exp(-((np.sqrt(X*X+Y*Y)-.58)/.14)**2); noise=np.random.default_rng(19).normal(size=signal.shape)
 field=np.sqrt(b)*(2*signal-1)+np.sqrt(1-b)*noise
 for i in range(16):
  for j in range(16):
   v=float(np.clip((field[i,j]+1)/2,0,1)); color=tuple((1-v)*np.ones(3)+v*np.array(matplotlib.colors.to_rgb(T)))
   a.add_patch(Rectangle((x-.65+j*.08125,1.25+i*.08125),.082,.082,fc=color,ec='none'))
 tx(a,x,.72,label,18)
for l,r in zip(xs[:-1],xs[1:]):
 ar(a,l+.65,2.94,r-.65,2.94,G); ar(a,r-.65,1.0,l+.65,1.0,T)
tx(a,2.4,2.94,r'$\cdots$',13,G,bbox={'facecolor':'white','edgecolor':'none','pad':1}); tx(a,7.6,2.94,r'$\cdots$',13,G,bbox={'facecolor':'white','edgecolor':'none','pad':1})
tx(a,5,3.4,'Forward process  $q$  ·  fixed',11,G)
tx(a,5,.07,r'Reverse process  $p_\theta$  ·  learned',11,T)
tx(a,1.1,.39,'Data',9,G); tx(a,8.9,.39,'Gaussian noise',9,G)
save(f,'02-diffusion-chain')
# 3: analytic decomposition, no invented numerical bars.
f,a=base(4.5)
tx(a,5,4.09,r'$\mathcal{L}_{\mathrm{VLB}}=\mathbb{E}_{q}\!\left[L_T+\sum_{t=2}^{T}L_{t-1}+L_0\right]$',24)
xx=[1.65,5,8.35]
for x in xx: a.plot([x-1.35,x+1.35],[3.58,3.58],c=L,lw=1)
tx(a,xx[0],3.2,'Prior matching',11,G); tx(a,xx[1],3.2,'Denoising',11,T); tx(a,xx[2],3.2,'Reconstruction',11,O)
tx(a,xx[0],2.65,r'$D_{\mathrm{KL}}\!\left(q(x_T\mid x_0)\,\|\,p(x_T)\right)$',14)
tx(a,xx[1],2.65,r'$D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\right)$',12)
tx(a,xx[2],2.65,r'$-\log p_\theta(x_0\mid x_1)$',15)
tx(a,xx[0],2.18,'Fixed noise schedule',9,G); tx(a,xx[2],2.18,'Observation model',9,G)
ar(a,5,2.28,5,1.75,T); tx(a,5,1.43,r'$w_t\,\|\epsilon-\epsilon_\theta(x_t,t)\|^2+C$',21,T)
tx(a,7.45,1.94,'Fixed reverse variance',9,G)
ar(a,5,1.13,5,.7,O); tx(a,7.65,.91,'Reweight timesteps',10,O)
tx(a,5,.39,r'$\mathcal{L}_{\mathrm{simple}}=\mathbb{E}_{t,x_0,\epsilon}\,\|\epsilon-\epsilon_\theta(x_t,t)\|^2$',21)
save(f,'03-elbo-pieces')
# 4: temporal stacks are vectors; upper autoencoder / lower latent sampling.
def frames(a,x,y,c=N,latent=False):
 for i in range(3):
  dx=.11*i; dy=.12*i; a.add_patch(Rectangle((x-.47+dx,y-.33+dy),.75,.61,fc='white',ec=c,lw=1))
  if latent:
   for k in range(3):
    for j in range(2): a.add_patch(Rectangle((x-.42+dx+k*.21,y-.25+dy+j*.22),.13,.13,fc=c,ec='none',alpha=.25+.2*((k+j)%2)))
  else:
   a.plot([x-.4+dx,x-.19+dx,x+.02+dx,x+.22+dx],[y-.24+dy,y+.01+dy,y-.1+dy,y+.15+dy],c=c,lw=.9)
f,a=base(4.5)
tx(a,.2,4.15,'a',12,ha='left',fontweight='bold'); tx(a,.55,4.15,'Representation',11,ha='left')
frames(a,.85,3.18,N); tx(a,.95,2.46,'Video $x$',11); ar(a,1.48,3.25,2.13,3.25)
mod(a,2.75,3.25,.9,1.05,'$E$',T); ar(a,3.28,3.25,3.98,3.25); frames(a,4.7,3.18,T,True); tx(a,4.8,2.46,'Latent $z_0$',11)
ar(a,5.38,3.25,6.12,3.25); mod(a,6.75,3.25,.9,1.05,'$D$',T,False); ar(a,7.28,3.25,8.23,3.25); frames(a,9,3.18,N); tx(a,9.1,2.46,'Reconstruction',11); ar(a,.43,3.84,.93,3.84,G); tx(a,1.52,3.84,'Frame $f$',9,G)
a.plot([.2,9.8],[2.08,2.08],c=L,lw=.8)
tx(a,.2,1.74,'b',12,ha='left',fontweight='bold'); tx(a,.55,1.74,'Generation',11,ha='left')
tx(a,.85,.99,'$z_K$',19,O); ar(a,1.25,.99,1.95,.99,O); tx(a,3.05,.99,r'$z_{K-1}\;\cdots\;z_1$',20); ar(a,4.17,.99,4.48,.99,T); tx(a,4.95,.99,'$z_0$',19,T)
ar(a,5.35,.99,6.12,.99); mod(a,6.75,.99,.9,1.05,'$D$',T,False); ar(a,7.28,.99,8.23,.99); frames(a,9,.91,N); tx(a,9.1,.21,'New video',11)
tx(a,.85,.4,'Noise',10,O); tx(a,3.1,.39,'Denoising step $k$',10,G)
save(f,'04-latent-video')
print('Saved four vector SVG/PDF figures and PNG previews to',P)
