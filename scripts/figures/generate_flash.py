"""Original conceptual diagrams. No plotted values are experimental measurements."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

NAVY='#243746'; TEAL='#267D8D'; ORANGE='#C77C46'; GREY='#84939C'; LIGHT='#DDE4E7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':NAVY,'axes.labelcolor':NAVY,'mathtext.fontset':'stix','svg.fonttype':'path','svg.hashsalt':'flash-grpo-figures-v1','pdf.fonttype':42,'axes.linewidth':.7,'savefig.facecolor':'white'})

def arrow(ax,a,b,color=NAVY,lw=1.3,style='-|>',ms=10):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=ms,lw=lw,color=color,shrinkA=0,shrinkB=0))

def save(fig,name,out):
    for fmt in ('svg','pdf','png'):
        fig.savefig(out/f'{name}.{fmt}',dpi=200,bbox_inches='tight',pad_inches=.16,metadata=({'Creator':'Python / Matplotlib','Date':None} if fmt=='svg' else {'Creator':'Python / Matplotlib','CreationDate':None,'ModDate':None} if fmt=='pdf' else {}))
    plt.close(fig)

def rollout(out):
    fig,ax=plt.subplots(figsize=(8.6,3.75));ax.set(xlim=(-1.8,8.3),ylim=(-.75,3.05));ax.axis('off')
    xs=np.arange(0,7)
    for y,title,selected in [(2.0,'Multi-step',range(6)),(.35,'Flash-GRPO',[3])]:
        ax.text(-1.65,y,title,va='center',weight='medium',fontsize=11)
        for k in range(6):
            color=TEAL if (title=='Flash-GRPO' and k!=3) else NAVY
            arrow(ax,(xs[k]+.12,y),(xs[k+1]-.12,y),color=color,lw=1.3,ms=8)
        ax.scatter(xs,np.full(7,y),s=30,facecolors='white',edgecolors=NAVY,lw=1.2,zorder=5)
        for k in selected:
            ax.plot([k+.5,k+.5],[y-.08,y-.36],color=ORANGE,lw=1)
            ax.scatter(k+.5,y-.43,s=30,marker='s',color=ORANGE,zorder=5)
        arrow(ax,(6.18,y),(6.9,y),GREY,ms=8)
        ax.text(7.1,y,'Video',va='center',fontsize=10)
        ax.text(7.1,y-.30,r'$R$',color=TEAL,fontsize=12)
    ax.text(0,2.4,r'$x_T$',ha='center',fontsize=13);ax.text(6,2.4,r'$x_0$',ha='center',fontsize=13)
    ax.text(3,2.72,'Full denoising rollout',ha='center',fontsize=11)
    ax.text(3.5,.68,'SDE',ha='center',fontsize=9,color=ORANGE)
    ax.text(1.5,.68,'ODE',ha='center',fontsize=9,color=TEAL)
    ax.text(5.2,.68,'ODE',ha='center',fontsize=9,color=TEAL)
    ax.plot([-1.65,8],[1.1,1.1],color=LIGHT,lw=.7)
    ax.scatter(.2,-.62,s=28,marker='s',color=ORANGE);ax.text(.43,-.62,'Policy-loss term',va='center',fontsize=9)
    ax.text(6.1,-.62,'Positions are schematic',ha='center',fontsize=8,color=GREY)
    save(fig,'rollout-vs-update',out)

def grouping(out):
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.35),gridspec_kw={'wspace':.22})
    for ai,(ax,ttl,pos) in enumerate(zip(axes,['Independent time selection','Iso-temporal grouping'],[[1,3,4,2],[3,3,3,3]])):
        ax.set(xlim=(-.7,6.8),ylim=(-.7,4.1));ax.axis('off')
        ax.text(-.55,3.85,f'({chr(97+ai)})',fontsize=10,weight='bold')
        ax.text(.15,3.85,ttl,fontsize=10,weight='medium')
        if ai==1:
            ax.axvspan(2.85,3.15,ymin=.19,ymax=.81,color=ORANGE,alpha=.10,lw=0)
            ax.plot([3,3],[.05,3.35],color=ORANGE,lw=.8,ls=(0,(2,3)))
            ax.text(3,3.47,r'$t_k$',ha='center',color=ORANGE,fontsize=12)
        for i,p in enumerate(pos):
            y=3-i*.85
            ax.text(-.4,y,rf'$z_{{{i+1}}}$',ha='center',va='center',fontsize=11,color=GREY)
            arrow(ax,(0,y),(5.6,y),GREY,lw=1.0,ms=7)
            ax.plot([p-.28,p+.28],[y,y],color=ORANGE,lw=3,solid_capstyle='round')
            ax.scatter([p-.28,p+.28],[y,y],s=16,color='white',edgecolor=ORANGE,lw=1,zorder=3)
            ax.text(5.92,y,rf'$R_{i+1}$',va='center',fontsize=11,color=TEAL)
        ax.text(2.75,-.22,'Diffusion time  '+r'$T\ \longrightarrow\ 0$',ha='center',fontsize=9,color=GREY)
    fig.text(.5,.015,'One prompt per group · independent initial noise · shared SDE time in ITG',ha='center',fontsize=9,color=NAVY)
    save(fig,'iso-temporal-grouping',out)

def rectification(out):
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.5),gridspec_kw={'wspace':.32})
    # Illustrative vectors only: rectification preserves each direction but not equal norms.
    h=np.array([[.75,.5],[-.3,.85],[.55,-.3]])
    lambdas=np.array([1.7,.65,1.25]); cols=[NAVY,TEAL,ORANGE]
    for j,ax in enumerate(axes):
        ax.set(xlim=(-.8,1.55),ylim=(-.65,1.28),aspect='equal');ax.axis('off')
        arrow(ax,(-.62,0),(1.4,0),LIGHT,lw=.8,ms=6);arrow(ax,(0,-.5),(0,1.02),LIGHT,lw=.8,ms=6)
        ax.scatter(0,0,s=16,color=NAVY,zorder=5)
        for i,(v,l,col) in enumerate(zip(h,lambdas,cols)):
            v=v*(l if j==0 else 1)
            arrow(ax,(0,0),v,col,lw=2,ms=11)
            ax.text(v[0]+(-.06 if v[0]<0 else .07),v[1]+(.06 if v[1]>0 else -.06),rf'$t_{i+1}$',color=col,ha='right' if v[0]<0 else 'left',va='center',fontsize=11)
        ax.text(.35,1.18,r'$g_t=\lambda(t)\,h_t$' if j==0 else r'$\widetilde{g}_t=h_t$',ha='center',fontsize=17)
    fig.text(.5,.52,r'$\times\ \lambda(t)^{-1}$',ha='center',fontsize=13,color=GREY)
    fig.text(.5,.075,'Explicit time scale removed; directions and norms still differ.',ha='center',fontsize=10)
    fig.text(.5,.015,'Schematic, unclipped policy term · hₜ retains sample- and time-dependent factors',ha='center',fontsize=8,color=GREY)
    save(fig,'temporal-gradient-rectification',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path(__file__).parent);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    rollout(a.output_dir);grouping(a.output_dir);rectification(a.output_dir)
