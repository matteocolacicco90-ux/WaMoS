#!/usr/bin/env python
# coding: utf-8
from hapin import*
#from pylab import plot
#import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import lmfit as lm
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.integrate import trapezoid
import sys
import subprocess
import os

# In[2]:


db_begin('data/CO-data')


# In[3]:


od_nu=4284
do_nu=4286

nu_res=0.001

OmegaGrid = np.arange(od_nu, do_nu, nu_res)

print(OmegaGrid)

ela = 10

volfr_h2o=0.18
volfr_n2= 0.71
volfr_co2=0.10
volfr_co= 0.02

print(ela)

# --- Načtení teploty z argumentu ---
if len(sys.argv) < 3:
    print("Použití: python fit2voigt.py dummy_label T_env")
    sys.exit(1)
    
T_env = float(sys.argv[2])

P_env = 1
#T_env = 2400
P_ref = 1
T_ref = 296
#p_self=1
#p_self = volfr_h2o*P_env
p_self = volfr_co*P_env


NSL_min = 1.0e-23

print(P_env,T_env,P_ref,T_ref,p_self)

cond = ('AND', ('BETWEEN','nu',od_nu,do_nu),('>=','sw',1e-45))

#select('23um01_HITEMP2010', ParameterNames=('nu','sw','delta_air','gamma_air','gamma_self','molec_id','local_iso_id','elower','n_air'),Conditions=cond,DestinationTableName='H2O-select')
select('23um05_HITEMP2019', Conditions=cond,DestinationTableName='CO-select')

print(LOCAL_TABLE_CACHE['CO-select']['data'].keys())

nue,sw,delta_air,gamma_air,gamma_self,molec_id,local_iso_id,elower,n_air = getColumns('CO-select',['nu','sw','delta_air','gamma_air','gamma_self','molec_id','local_iso_id','elower','n_air'])


# In[4]:


nu, coef_HT = absorptionCoefficient_HT(
    SourceTables='CO-select',
    Environment={'T': T_env, 'p': P_env},
    OmegaGrid=OmegaGrid,
    Diluent={'self': volfr_co,'H2O': volfr_h2o, 'air': volfr_n2, 'CO2': volfr_co2},
    #HITRAN_units=False
)
nu, coef_Voigt = absorptionCoefficient_Voigt(
    SourceTables='CO-select',
    Environment={'T': T_env, 'p': P_env},
    OmegaGrid=OmegaGrid,
#    Diluent={'self': volfr_co,'H2O': volfr_h2o, 'air': volfr_n2, 'CO2': volfr_co2}
     Diluent={'self': volfr_co, 'air': 1.0}
)

# In[5]:


#plt.plot(nu,coef_HT,'orange',nu, coef_Voigt,'lime') # plot both profiles
#plt.xlim(od_nu, do_nu)
#plt.plot(nu,coef_HT-coef_Voigt)
#plt.show()


# In[6]:


sum_lor=np.zeros(len(nu))
sum_voi=np.zeros(len(nu))

lor=[np.zeros(len(nu))]*len(nue)
voi=[np.zeros(len(nu))]*len(nue)

nux=[np.zeros(len(nu))]*len(nue)
g2f=[np.zeros(len(nu))]*len(nue)

delta=np.zeros(len(nue))
gammaD=np.zeros(len(nue))
gammaL=np.zeros(len(nue))
gammaLD=np.zeros(len(nue))
gammaLD1=np.zeros(len(nue))
gammaLD2=np.zeros(len(nue))
n_self=np.zeros(len(nue))

sw_env=np.zeros(len(nue))
nsl=np.zeros(len(nue))
vline=np.zeros(len(nue))
dvline=np.zeros(len(nue))
a0=np.zeros(len(nue))
#mW=np.ones(len(nue))

#I0=np.ones(len(nue))
#I0 = []
#for i in range(len(nue)):
#    I0.append(PowerFit(nue[i]))


# In[7]:


AvogN = 6.02214129e23
c2=1.43877
fSqrtMass = sqrt(molecularMass(molec_id[0],local_iso_id[0]))

Q_env = partitionSum(molec_id[0],local_iso_id[0],T_env)
Q_ref = partitionSum(molec_id[0],local_iso_id[0],T_ref)

print(Q_env, Q_ref)

n=-1

molec2atm = 7.34e21/T_env

#GammaCon = 0.155/0.04
#GammaCon = 1.0


# In[8]:


for i in range(len(nue)):
    sw_env[i] =  sw[i]*(Q_ref/Q_env)*(np.exp(-c2*elower[i]/T_env)/np.exp(-c2*elower[i]/T_ref))*((1-np.exp(-c2*nue[i]/T_env))/(1-np.exp(-c2*nue[i]/T_ref)))

    delta[i] = delta_air[i]*(P_env - p_self)
    gammaD[i] = nue[i]*(cSqrt2Ln2/cc)*sqrt(AvogN*cBolts)*sqrt(T_env)/fSqrtMass

    #n_self[i] = 0.7
    n_self[i] = n_air[i]

    #gamma_air[i] = GammaCon*gamma_air[i]

    gammaL[i] = (P_env-p_self)*gamma_air[i]*(T_ref/T_env)**n_air[i] + p_self*gamma_self[i]*(T_ref/T_env)**n_self[i]

#def PROFILE_VOIGT(Nu,GammaD,Gamma0,Delta0,WnGrid,YRosen=0.0,Sw=1.0): 
    # Voigt profile based on HTP.
    # Input parameters:
    #      Nu        : Unperturbed line position in cm-1 (Input).
    #      GammaD    : Doppler HWHM in cm-1 (Input)
    #      Gamma0    : Speed-averaged line-width in cm-1 (Input).       
    #      Delta0    : Speed-averaged line-shift in cm-1 (Input).
    #      WnGrid    : Current WaveNumber of the Computation in cm-1 (Input).
    #      YRosen    : 1st order (Rosenkranz) line mixing coefficients in cm-1 (Input)

    voi[i] = PROFILE_VOIGT(nue[i],gammaD[i],gammaL[i],delta[i],nu) # calc Voigt

    #gammaLD1[i] = (gammaL[i]+np.sqrt(gammaL[i]**2 + (gammaD[i]/2)**2))/2
    #gammaLD1[i] = (gammaL[i]/2+np.sqrt((gammaL[i]/4)**2 + gammaD[i]**2))/2
    #gammaLD2[i] = np.sqrt(gammaL[i]**2 + gammaD[i]**2)
    #gammaLD[i] = (gammaLD1[i] + gammaLD2[i])/2

    gammaLD[i]= (0.5346*gammaL[i] + np.sqrt(0.2166*gammaL[i]**2 + 8*np.log(2)*gammaD[i]**2))/2

    #lor[i] = PROFILE_LORENTZ(nue[i],gammaLD[i],delta[i],nu) # calc Lorentz
    lor[i] = PROFILE_LORENTZ(nue[i]+delta[i],gammaLD[i],delta[i],nu) # calc Lorentz

    nsl[i] = sw_env[i]*p_self*ela
    print(nue[i]+delta[i], sw_env[i], gammaL[i], gammaD[i], T_env, P_env)

    if nsl[i] > NSL_min:
        n=n+1
        sum_voi = sum_voi + voi[i]*sw_env[i]
        sum_lor = sum_lor + lor[i]*sw_env[i]
        vline[n] = nue[i] + delta[i]
        dvline[n] = gammaLD[i]
        elower[n] = elower[i]
#        a0[n] = nsl[i]*molec2atm/(np.pi*dvline[n])
        a0[n] = nsl[i]*molec2atm

#        mW[n] = I0[i]
        #print("i=", i, nue[i]+delta[i], nsl[i], gammaLD[i], T_env, P_env)
#        print("n=", n, mW[n], vline[n], elower[n], a0[n], dvline[n], T_env, P_env) 
        print("n=", n, vline[n], elower[n], a0[n], dvline[n], T_env, P_env)


# In[9]:


#Multi-line model:
coef_ela=coef_Voigt * molec2atm*p_self*ela

#Two-line models:
my_voi_atm = sum_voi * molec2atm*p_self*ela
my_lor_atm = sum_lor * molec2atm*p_self*ela

diff_voi = my_voi_atm-coef_ela
reldiff_voi = diff_voi # calc difference

diff_lor = my_lor_atm-(coef_ela)
reldiff_lor = 100*diff_lor/(coef_ela)   # calc difference

"""
plt.subplot(2,1,1) # upper panel
#plt.plot(nu,coef_air*molec2atm*p_self,'orange',nu,sum_voi_atm,'red') # plot both profiles
plt.plot(nu,coef_ela,'orange',nu,my_voi_atm,'red',nu,my_lor_atm,'blue') # plot both profiles
plt.xlim(od_nu+0.5, do_nu-0.5)
plt.legend(['HAPI-Voigt','MY-Voigt','MY-Lorentz']) # show legend
plt.title('Voigt and Lorentz profiles') # show title
plt.subplot(2,1,2) # lower panel
plt.xlim(od_nu+0.5, do_nu-0.5)
#plt.plot(nu, coef_HT*molec2atm*p_self*ela-coef_Voigt*molec2atm*p_self*ela,'orange') # plot difference
#plt.title('HT-hapin vs. MY-difference') # show title
plt.plot(nu, diff_voi,'red') # plot difference
plt.legend(['MY-HAPI diff.']) # show legend
#plt.title('HT-hapin vs. MY-difference') # show title
plt.show() # show all figures
"""

# In[10]:


line1 = [vline[0], a0[0]]
line2 = [vline[1], a0[1]]

x_wn=nu
y=coef_ela

c1_fixed, c2_fixed = line1[0], line2[0]
a1_guess, a2_guess = line1[1], line2[1]

sigma_theo = (1/2*(c1_fixed+c2_fixed))*(cSqrt2Ln2/cc)*sqrt(AvogN*cBolts)*sqrt(T_env)/fSqrtMass

# ============================================
# Part 1: Load data, define Voigt profile, model
# ============================================

# --- Voigt profile ---
def voigt_wn(x, center, amplitude, sigma, gamma):
    z = ((x - center) + 1j*gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

def softplus(w): return np.log1p(np.exp(-np.abs(w)))+np.maximum(w,0)
def inv_softplus(target): return np.log(np.expm1(max(target,1e-12)))
def softshift(raw, limit=0.01): return limit*np.tanh(raw)

# --- Model with shared sigma + soft-bounded shifts ---
def two_voigt_shared_sigma_shift_soft(x, a1, g1_raw, a2, g2_raw, s_raw, d1_raw, d2_raw):
    g1 = softplus(g1_raw); g2 = softplus(g2_raw); sigma = softplus(s_raw)
    d1 = softshift(d1_raw,0.01); d2 = softshift(d2_raw,0.01)
    c1 = c1_fixed+d1; c2 = c2_fixed+d2
    return voigt_wn(x,c1,a1,sigma,g1)+voigt_wn(x,c2,a2,sigma,g2)

# ============================================
# Part 2: Fit parameters and find inflection points
# ============================================

# --- Fit ---
p0=[a1_guess,inv_softplus(0.02),a2_guess,inv_softplus(0.02),inv_softplus(0.03),0.0,0.0]
popt,_=curve_fit(two_voigt_shared_sigma_shift_soft,x_wn,y,p0=p0,maxfev=30000)

a1,g1_raw,a2,g2_raw,s_raw,d1_raw,d2_raw=popt
sigma=softplus(s_raw); g1=softplus(g1_raw); g2=softplus(g2_raw)
d1=softshift(d1_raw,0.01); d2=softshift(d2_raw,0.01)
c1_fit=c1_fixed+d1; c2_fit=c2_fixed+d2

print(f"Theoretical Doppler broadening HWHM: sigma= {sigma_theo:.3e}")
print(f"a1={a1:.3e}, a2={a2:.3e}, sigma={sigma:.3e}, g1={g1:.3e}, g2={g2:.3e}")
print(f"centers: c1={c1_fit:.5f}, c2={c2_fit:.5f}")

# --- Profiles ---
y_total=two_voigt_shared_sigma_shift_soft(x_wn,*popt)
y1=voigt_wn(x_wn,c1_fit,a1,sigma,g1)
y2=voigt_wn(x_wn,c2_fit,a2,sigma,g2)

# --- Inflection points ---
def inflection_bounds(x,y,center):
    dy=np.gradient(y,x); d2y=np.gradient(dy,x)
    i0=np.argmin(np.abs(x-center))
    iL,iR=None,None
    for i in range(i0-1,1,-1):
        if np.sign(d2y[i])!=np.sign(d2y[i-1]): iL=i; break
    for i in range(i0+1,len(x)-1):
        if np.sign(d2y[i])!=np.sign(d2y[i-1]): iR=i; break
    def refine(i):
        x1,x2=x[i-1],x[i]; y1_,y2_=d2y[i-1],d2y[i]
        t=-y1_/(y2_-y1_) if (y2_-y1_)!=0 else 0.5
        return x1+t*(x2-x1)
    xL=refine(iL) if iL else x[i0-5]; xR=refine(iR) if iR else x[i0+5]
    return xL,xR

xL1,xR1=inflection_bounds(x_wn,y1,c1_fit)
xL2,xR2=inflection_bounds(x_wn,y2,c2_fit)

print(f"Line1 inflection bounds: {xL1:.5f}, {xR1:.5f}")
print(f"Line2 inflection bounds: {xL2:.5f}, {xR2:.5f}")

# --- FWHM and integrals ---
def voigt_fwhm_cm1(sigma, gamma):
    return 0.5346*gamma + np.sqrt(0.2166*gamma**2 + 8*np.log(2)*sigma**2)
#note from https://www.mdpi.com/2571-6182/7/2/23
#(𝛾_𝐿 and 𝛾_𝐺 are the HWHM (half with at half maximum) values of the Lorentzian and Gaussian profiles, respectively
# Approximation with an accuracy of 0.02% was given by Olivero et al. [51] 
#(originally found by Kielkopf [52]):𝛾_V≈0.5346𝛾_L+√0.2166𝛾2_L+𝛾2_G(35)
#Expression (35) is also exact for a pure Gaussian or Lorentzian.    

def line_integral(center, a, s, g):
    fwhm = voigt_fwhm_cm1(s, g)
    x_int = np.linspace(center - fwhm/2, center + fwhm/2, 600)
    y_int = voigt_wn(x_int, center, a, s, g)
    area = trapezoid(y_int, x_int)
    return fwhm, area, x_int, y_int

fwhm1, area1, x1_int, y1_int = line_integral(c1_fit, a1, sigma, g1)
fwhm2, area2, x2_int, y2_int = line_integral(c2_fit, a2, sigma, g2)

print(f"Line1 FWHM={fwhm1:.3e} cm^-1, Integral={area1:.3e}")
print(f"Line2 FWHM={fwhm2:.3e} cm^-1, Integral={area2:.3e}")


# In[11]:


# ============================================
# Part 3a: Compute integrals and ratios
# ============================================

def area_inflection_total_and_trimmed(center, a, s, g, xL, xR, npts=800):
    # Zajisti správné pořadí inflexních bodů
    if xL > xR:
        xL, xR = xR, xL

    xi = np.linspace(xL, xR, npts)
    yi = voigt_wn(xi, center, a, s, g)

    # Celková plocha pod křivkou
    area_total = trapezoid(yi, xi)

    # Baseline = min hodnoty profilu v inflexních bodech
    yL = voigt_wn(xL, center, a, s, g)
    yR = voigt_wn(xR, center, a, s, g)
    baseline = min(yL, yR)

    # Odečtení obdélníku pod baseline
    rect = (xR - xL) * baseline
    trimmed_area = area_total - rect

    # Pojistka proti záporným hodnotám
    eps = 1e-30
    area_total = max(area_total, eps)
    trimmed_area = max(trimmed_area, eps)

    return area_total, trimmed_area, xi, yi, baseline

area1_total, area1_trim, xi1, yi1, base1 = area_inflection_total_and_trimmed(c1_fit, a1, sigma, g1, xL1, xR1)
area2_total, area2_trim, xi2, yi2, base2 = area_inflection_total_and_trimmed(c2_fit, a2, sigma, g2, xL2, xR2)

# Bezpečné poměry
ratio_db = a1_guess / a2_guess if a2_guess > 0 else float('nan')
ratio_amp = a1 / a2 if a2 > 0 else float('nan')
ratio_trim = area1_trim / area2_trim if area2_trim > 0 else float('nan')
ratio_total = area1_total / area2_total if area2_total > 0 else float('nan')

print(f"Ratios: database = {ratio_db:.6f}, amplitude = {ratio_amp:.6f}")
print(f"Ratios: total = {ratio_total:.6f}, trimmed = {ratio_trim:.6f}")

print(f"Line1 area_total = {area1_total:.3e}, trimmed = {area1_trim:.3e}")
print(f"Line2 area_total = {area2_total:.3e}, trimmed = {area2_trim:.3e}")

import os

# --- Výstup poměrů ---
cats = ['HITEMP intensity ratio',
        'Amplitude ratio',
        'Total-area ratio',
        'Trimmed-area ratio']

vals = [a1_guess / max(a2_guess, 1e-30),
        a1 / max(a2, 1e-30),
        area1_total / max(area2_total, 1e-30),
        area1_trim / max(area2_trim, 1e-30)]

# --- Přidání do CSV ---
csv_file = "line_ratios_vs_temperature.csv"
header = "T_env;HITEMP_intensity_ratio;Amplitude_ratio;Total_area_ratio;Trimmed_area_ratio\n"

# Pokud soubor neexistuje, vytvoří se s hlavičkou
if not os.path.exists(csv_file):
    with open(csv_file, "w") as f:
        f.write(header)

# Přidání řádku s výsledky
with open(csv_file, "a") as f:
    f.write(f"{T_env};{vals[0]:.6f};{vals[1]:.6f};{vals[2]:.6f};{vals[3]:.6f}\n")
