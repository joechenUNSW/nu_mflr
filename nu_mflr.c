#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>

#include "pcu.h"

////////////////////////////////// CONSTANTS ///////////////////////////////////
//All dimensionful quantities are in units of Mpc/h to the appropriate power,
//unless otherwise noted.  Values declared as const double or const int  may be
//modified by the user, while those in #define statements are derived parameters
//which should not be changed.

//conformal hubble today
const double Hc0h = 3.33564095198152e-04; //(1e2/299792.458)
#define Hc0h2 (Hc0h*Hc0h)

//initial scale factor, and max value of eta=ln(a/a_in)
const double aeta_in = 1e-3; 
#define eta_stop (-log(aeta_in))

//density fractions today; assume flat universe with CB, nu, photons, Lambda
const double h = 0.6724; //H_0 / (100 km/sec/Mpc)
const double T_CMB_0_K = 2.726; //CMB temperature today, in Kelvins

const double Omega_cb_0 = 0.3121; //CDM+Baryon density fraction today
const double Omega_nu_0 = 0.00713171; //massive nu density fraction today

const double N_nu_eff = 3.044; //effective number of neutrinos in early univ
const double N_nu_massive = 3.044; //number of massive neutrinos
const int N_tau = 20; //number of neutrino streams; maximum 900 for this pcu.h
const int N_mu = 20; //number of multipoles to track for each stream

#define m_nu_eV (93.259*Omega_nu_0*h*h/N_nu_massive)
#define Omega_nu_t_0 (Omega_nu_0/N_tau)

#define T_CMB_0_K_4 (T_CMB_0_K*T_CMB_0_K*T_CMB_0_K*T_CMB_0_K)
#define Omega_gam_0 ((4.46911743913795e-07)*T_CMB_0_K_4/(h*h))
#define Omega_nurel_0 (0.227107317660239*(N_nu_eff-N_nu_massive)*Omega_gam_0)
#define Omega_nugam_0 ((1.0+0.227107317660239*N_nu_eff)*Omega_gam_0)
#define Omega_rel_0 (Omega_gam_0+Omega_nurel_0)
#define Omega_de_0 (1.0-Omega_cb_0-Omega_nu_0-Omega_rel_0)

//code switches and parameters
const int SWITCH_OUTPUT_ALLFLUIDS = 0; //output all nu,cb perturbations
const int SWITCH_OUTPUT_MONO = 0; //output only monopoles (d_cdm,t_cdm,d_nu)
const double PARAM_DETA0 = 1e-6; //default starting step size in eta
const double PARAM_EABS = 0; //absolute error tolerance
const double PARAM_EREL = 1e-6; //relative error tolerance
const int DEBUG_NU_MOMENTA = 1;

//////////////////////////////////// NEUTRINOS /////////////////////////////////

//total number of equations:
//  2*N_tau*N_mu (delta and theta for each of N_tau nu streams and N_mu moments)
//  + 2 (delta and theta for CDM+Baryons)
#define N_EQ (2*N_tau*N_mu+2)

//homogeneous-universe momentum [eV], used to identify neutrino streams
const int FREE_TAU_TABLE = -4375643; //some negative integer, pick any
double tau_t_eV(int t){

  if(N_tau==0) return 0.0;
  static int init = 0;
  static double *tau_table_eV;

  if(!init){
    tau_table_eV = malloc(N_tau * sizeof(double));
    gsl_interp_accel *spline_accel = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_steffen,pcu_N);
    gsl_spline_init(spline,pcu_prob,pcu_tau,pcu_N);

    if(DEBUG_NU_MOMENTA) printf("#tau_t_eV: momenta [eV]:");
    
    for(int t=0; t<N_tau; t++){
      double prob = (0.5+t) / N_tau;
      tau_table_eV[t] = gsl_spline_eval(spline,prob,spline_accel);
      if(DEBUG_NU_MOMENTA) printf(" %g",tau_table_eV[t]);
    }

    if(DEBUG_NU_MOMENTA) printf("\n");
    gsl_spline_free(spline);
    gsl_interp_accel_free(spline_accel);
    init = 1;
  }

  if(t == FREE_TAU_TABLE){
    free(tau_table_eV);
    init = 0;
    return 0;
  }
  return tau_table_eV[t];
}

//speed -tau_t / tau0_t of each neutrino species
double v_t_eta(int t, double eta){
  double t_ma = tau_t_eV(t) / ( m_nu_eV * aeta_in*exp(eta) );
  return (t_ma<1 ? t_ma : 1);
}

double v2_t_eta(int t, double eta){
  double vt = v_t_eta(t,eta);
  return vt*vt;
}

//density ratio rho_t(eta)/rho_t(eta_stop) * aeta^2 and its log deriv
double Ec_t_eta(int t, double eta){ return 1.0/ ( aeta_in*exp(eta) ); }

double dlnEc_t_eta(int t, double eta){ return -1.0; }

//relativistic versions of the above, for Hubble rate calculation
double v2_t_eta_REL(int t, double eta){
  double m_aeta_tau = m_nu_eV * aeta_in*exp(eta) / tau_t_eV(t);
  return 1.0 / (1.0 + m_aeta_tau*m_aeta_tau);
}

double v_t_eta_REL(int t, double eta){ return sqrt(v2_t_eta_REL(t,eta)); }

double Ec_t_eta_REL(int t, double eta){
  double vt2 = v2_t_eta_REL(t,eta), aeta = aeta_in*exp(eta);
  if(1-vt2 < 1e-12){
    double ma_tau = m_nu_eV * aeta / tau_t_eV(t);
    return sqrt(1.0 + ma_tau*ma_tau) / (aeta*ma_tau); 
  }
  return 1.0 / (aeta * sqrt(1.0 - vt2));
}

double dlnEc_t_eta_REL(int t, double eta){ return -1.0 - v2_t_eta_REL(t,eta);}

//////////////////////////// HOMOGENEOUS COSMOLOGY /////////////////////////////

//equations of state: cdm, photons, DE
const double w_eos_cdm = 0, w_eos_gam = 0.333333333333333,
  w0_eos_de = -1.0, wa_eos_de = 0.0;

//a(eta)^2 * rho_de(eta) / rho_de_0 and its derivative
double Ec_de_eta(double eta){
  double aeta = aeta_in * exp(eta);
  return pow(aeta,-1.0 - 3.0*(w0_eos_de + wa_eos_de)) *
    exp(3.0*wa_eos_de*(aeta-1.0));
}

double dlnEc_de_eta(double eta){
  double aeta = aeta_in * exp(eta);
  return -1.0 - 3.0*(w0_eos_de + wa_eos_de) + 3.0*wa_eos_de*aeta;
}

//conformal hubble parameter
double Hc2_Hc02_eta(double eta){

  //scale factor
  double aeta = aeta_in * exp(eta), aeta2 = aeta*aeta, Ec_de = Ec_de_eta(eta);

  //sum Omega_{t,0} aeta^2 rho_t(eta)/rho_t_0 over CDM, photons, and DE
  double sum_OEc = Omega_cb_0/aeta + Omega_rel_0/aeta2 + Omega_de_0*Ec_de;
  
  //neutrinos, using relativistic Omega_nu(eta)
  for(int t=0; t<N_tau; t++) sum_OEc += Omega_nu_t_0 * Ec_t_eta_REL(t,eta);

  return sum_OEc;
}

double Hc_eta(double eta){ return Hc0h * sqrt(Hc2_Hc02_eta(eta)); }

//d log(Hc) / d eta
double dlnHc_eta(double eta){
  
  double aeta = aeta_in*exp(eta), aeta2 = aeta*aeta;
  double pre = 1.0 / ( 2.0 * Hc2_Hc02_eta(eta) );
  
  double sum_OdEc = -(1.0 + 3.0*w_eos_cdm) *  Omega_cb_0/aeta //CDM
    - (1.0 + 3.0*w_eos_gam) * Omega_rel_0/aeta2 //photons + massless nu
    + dlnEc_de_eta(eta) * Omega_de_0 * Ec_de_eta(eta); //DE
  
  for(int t=0; t<N_tau; t++)//neutrino fluids
    sum_OdEc +=  dlnEc_t_eta_REL(t,eta) * Ec_t_eta_REL(t,eta) * Omega_nu_t_0;
  
  return pre * sum_OdEc;
}

//density fraction in spatially-flat universe
double OF_eta(int F, double eta){
  
  double Hc02_Hc2 = 1.0/Hc2_Hc02_eta(eta), aeta = aeta_in*exp(eta);

  if(F == N_tau) //CDM
    return Omega_cb_0 * pow(aeta,-1.0-3.0*w_eos_cdm) * Hc02_Hc2;
  else if(F == N_tau+1) //photons + massless nu
    return Omega_rel_0 * pow(aeta,-1.0-3.0*w_eos_gam) * Hc02_Hc2;
  else if(F == N_tau+2) //dark energy, assumed Lambda
    return Omega_de_0 * Ec_de_eta(eta) * Hc02_Hc2;
  else if(F<0 || F>N_tau+2) return 0.0; //no fluids should have these indices
  return Omega_nu_t_0 * Ec_t_eta(F,eta) * Hc02_Hc2;
}

//Poisson equation for Phi
double Poisson(double eta, double k, const double *y){
  double Hc2 = Hc0h2 * Hc2_Hc02_eta(eta), pre = -1.5 * Hc2 / (k*k);
  double sum_Od = OF_eta(N_tau,eta)*y[2*N_tau*N_mu];
  for(int t=0; t<N_tau; t++) sum_Od += OF_eta(t,eta) * y[2*t*N_mu];
  return pre * sum_Od;
}

//////////////////////////////// UTILITY FUNCTIONS /////////////////////////////

//minimum, maximum functions
inline double fmin(double x,double y){ return (x<y ? x : y); }
inline double fmax(double x,double y){ return (x>y ? x : y); }

//print all fluid perturbations
int print_results(double eta, const double *w){
  printf("%g",eta);
  for(int i=0; i<N_EQ; i++) printf(" %g",w[i]);
  printf("\n");
  return 0;
}

//neutrino density monopole from perturbation array
double d_nu_mono(double z, const double *y){
  
  double d_mono = 0, norm = 0, aeta = 1.0/(1.0+z), eta = log(aeta/aeta_in);
  
  for(int t=0; t<N_tau; t++){
    double E_m = 1.0;
    d_mono += y[2*t*N_mu] * E_m;
    norm += E_m;
  }

  return d_mono / norm;
} 

void d_nu_mono_stream(double z, const double *y){
  
  for(int t=0; t<N_tau; t++){
     double d_mono = y[2*t*N_mu];
     printf("%f,",d_mono);
  }
  printf("\n");
} 

//print CDM density/velocity monopoles and total neutrino density monopole
int print_mono(double eta, const double *w){
  double aeta = aeta_in*exp(eta), z = 1.0/aeta-1.0;
  printf("%g %g %g %g\n",eta,w[2*N_tau*N_mu],w[2*N_tau*N_mu+1],d_nu_mono(z,w));
  return 0;
}

/////////////////////////////// DERIVATIVES ////////////////////////////////////

//neutrino perturbation variables: X_{F,ell} with X of delta or theta,
//F referring to fluid (0 to N_tau-1 for nu; N_tau for CDM; etc.)
//and ell referring to Legendre coefficient (0 to N_mu-1)
//  y[0] = delta_{0,0} 
//  y[1] = theta_{0,0}
//  y[2] = delta_{0,1}
//  y[3] = theta_{0,1}
//      ...
//  y[2*N_mu+0] = delta_{1,0}
//  y[2*N_mu+1] = theta_{1,0}
//             ...
//  y[2*N_tau*N_mu - 2] = delta_{N_tau-1,N_mu-1}
//  y[2*N_tau*N_mu - 1] = theta_{N_tau-1,N_mu-1}
//
//cdm perturbation variables come after the neutrinos
//  y[2*N_tau*N_mu + 0] = delta_{CDM}
//  y[2*N_tau*N_mu + 1] = theta_{CDM}

double dn(int alpha, int ell, const double y[]){
  return (ell<0 ? 0 : y[2*alpha*N_mu + 2*ell]); }

double tn(int alpha, int ell, const double y[]){
  return (ell<0 ? 0 : y[2*alpha*N_mu + 2*ell + 1]); }

double gt(int alpha, int ell, const double y[]){
  return tn(alpha,ell,y)*(double)(ell*(ell-1))/(double)((2*ell-1)*(2*ell+1)); }

int der(double eta, const double *y, double *dy, void *par){

  //initialize
  double *pd = (double *)par, k = pd[0], k_H = k/Hc_eta(eta), k2_H2 = k_H*k_H,
    Phi = Poisson(eta,k,y), aeta = aeta_in*exp(eta), dlnHc = dlnHc_eta(eta);  

  //neutrino stream perturbations
  for(int t=0; t<N_tau; t++){

    double vt = v_t_eta(t,eta), kv_H = vt*k_H;
    
    //sum over Legendre moments of fluid equations, except the last two
    //which require approximation
    for(int ell=0; ell<N_mu-2; ell++){
      dy[2*t*N_mu + 2*ell]
	= kv_H * ( dn(t,ell-1,y) * ell / (2*ell-1)
                   - dn(t,ell+1,y) * (ell+1) / (2*ell+3) )
        + tn(t,ell,y);
      dy[2*t*N_mu + 2*ell + 1]
        = -(1.0 + dlnHc) * tn(t,ell,y)
        - k2_H2 * (ell==0) * Phi
        + kv_H * ( tn(t,ell-1,y) * ell / (2*ell-1)
                   - tn(t,ell+1,y) * (ell+1) / (2*ell+3) );
    }

    //special case for ell = ell_max-1 = N_mu-2
    int ell = N_mu-2;
    dy[2*t*N_mu + 2*ell]
      = kv_H * ( dn(t,ell-1,y) * (ell) / (2*ell-1)
                 - dn(t,ell+1,y) * (ell+1) / (2*ell+3) )
      + tn(t,ell,y);
    dy[2*t*N_mu + 2*ell + 1]
      = -(1.0 + dlnHc) * tn(t,ell,y)
      + kv_H * ( tn(t,ell-1,y) * (ell) / (2*ell-1)
                 - tn(t,ell+1,y) * (ell+1) / (2*ell+3) );
    
    //special case for ell = ell_max = N_mu-1
    ell = N_mu-1;
    dy[2*t*N_mu + 2*ell]
      = kv_H * ( dn(t,ell-1,y) / (2*ell-1)
                 - 3.0 * (ell) * dn(t,ell,y) / (2*ell+1)
                 + 4.0 * (ell-1) * dn(t,ell-1,y) / (2*ell-1)
                 - (ell-2) * dn(t,ell-2,y) / (2*ell-3) )
      + tn(t,ell,y);
    dy[2*t*N_mu + 2*ell + 1]
      = -(1.0 + dlnHc) * tn(t,ell,y)
      + kv_H * ( tn(t,ell-1,y) / (2*ell-1)
                 - 3.0 * (ell) * tn(t,ell,y) / (2*ell+1)
                 + 4.0 * (ell-1) * tn(t,ell-1,y) / (2*ell-1)
                 - (ell-2) * tn(t,ell-2,y) / (2*ell-3) );
  }

  //cdm perturbations
  dy[2*N_tau*N_mu + 0] = y[2*N_tau*N_mu + 1];
  dy[2*N_tau*N_mu + 1] = -(1.0 + dlnHc)*y[2*N_tau*N_mu + 1] -  k2_H2*Phi;
  
  return GSL_SUCCESS;
}

///////////////////////////////// EVOLUTION ////////////////////////////////////

//evolve from aeta_in to input redshift
int evolve_to_z(double k, double z, double *w){
  
  //initialize perturbations at eta=0
  double c_in=1, Omega_m_0 = Omega_cb_0+Omega_nu_0, fnu0 = Omega_nu_0/Omega_m_0;
  double aeta_eq = Omega_rel_0 / Omega_cb_0;
  for(int F=0; F<N_EQ; F++) w[F] = 0;

  //CDM+Baryon perturbations
  w[2*N_tau*N_mu + 0] = c_in * (aeta_in + (2.0/3.0)*aeta_eq);
  w[2*N_tau*N_mu + 1] = c_in * aeta_in;

  //neutrino perturbations: monopoles only
  for(int t=0; t<N_tau; t++){
    double m_t = m_nu_eV/tau_t_eV(t);
    double kfs2 = 1.5 * m_t*m_t * Hc0h2 * Omega_m_0 * aeta_in;
    double kfs = sqrt(kfs2), kpkfs = k + kfs, kpkfs2 = kpkfs*kpkfs;
    double Ft = (1.0-fnu0) * kfs2 / (kpkfs2 - fnu0*kfs2);
    double dlnFt = k*kpkfs / (kpkfs2 - fnu0*kfs2);
    w[2*t*N_mu + 0] = Ft * w[2*N_tau*N_mu + 0];
    w[2*t*N_mu + 1] = dlnFt*w[2*t*N_mu+0] + Ft*w[2*N_tau*N_mu+1];
  }
  
  //print initial conditions
  if(SWITCH_OUTPUT_ALLFLUIDS) print_results(0,w);
  else if(SWITCH_OUTPUT_MONO) print_mono(0,w);

  //initialize GSL ODE integration
  int status = GSL_SUCCESS;
  double eta0 = 0, aeta1 = 1.0/(1.0+z), eta1 = log(aeta1/aeta_in), par = k;

  gsl_odeiv2_system sys = {der, NULL, N_EQ, &par};
  
  gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys,
						       gsl_odeiv2_step_rkf45,
						       PARAM_DETA0,
						       PARAM_EABS,
						       PARAM_EREL);
  gsl_odeiv2_driver_set_hmax(d, 0.1);

  //integrate to input redshift, printing results at regular intervals
  double deta = 0.01, eta = eta0, etai = deta;

  while(etai < eta1
	&& status == GSL_SUCCESS){
    etai = fmin(eta+deta, eta1);
    status = gsl_odeiv2_driver_apply(d, &eta, etai, w);
    if(SWITCH_OUTPUT_ALLFLUIDS) print_results(eta,w);
    else if(SWITCH_OUTPUT_MONO) print_mono(eta,w);
  }

  //clean up and quit
  gsl_odeiv2_driver_free(d);
  return 0;
}

int evolve_step(double k, double z0, double z1, double *w){

  //initialize GSL ODE integration
  double aeta0 = 1.0/(1.0+z0), eta = log(aeta0/aeta_in), aeta1 = 1.0/(1.0+z1),
    eta1 = log(aeta1/aeta_in), par = k;

  gsl_odeiv2_system sys = {der, NULL, N_EQ, &par};
  
  gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys,
						       gsl_odeiv2_step_rkf45,
						       PARAM_DETA0,
						       PARAM_EABS,
						       PARAM_EREL);
  gsl_odeiv2_driver_set_hmax(d, 0.1);

  //integrate to final redshift and print results
  int status = gsl_odeiv2_driver_apply(d, &eta, eta1, w);
  if(SWITCH_OUTPUT_ALLFLUIDS) print_results(eta,w);
  else if(SWITCH_OUTPUT_MONO) print_mono(eta,w);

  //clean up and quit
  gsl_odeiv2_driver_free(d);
  return 0;
}

int compute_growths(int nz, const double *z_n, int nk, const double *k_n,
		    double *Dcb, double *fcb, double *Dnu, double *Tcb, double *dcb){

  int OUTPUT_STATUS = (N_tau*N_mu >= 1000); //for long jobs, print status
  
#pragma omp parallel for schedule(dynamic)
  for(int ik=0; ik<nk; ik++){
    double y[N_EQ], k=k_n[ik], aeta = 1.0/(1.0+z_n[0]), eta = log(aeta/aeta_in);
    evolve_to_z(k,z_n[0],y);
    
    Dcb[ik] = y[2*N_tau*N_mu];
    fcb[ik] = y[2*N_tau*N_mu+1]/y[2*N_tau*N_mu];
    Dnu[ik] = y[2*N_mu]; //d_nu_mono(z_n[0],y);
    Tcb[ik] = y[2*N_tau*N_mu+1];
    dcb[ik] = y[2*N_tau*N_mu];

    for(int iz=1; iz<nz; iz++){
      aeta = 1.0 / (1.0 + z_n[iz]);
      eta = log(aeta/aeta_in);
      evolve_step(k,z_n[iz-1],z_n[iz],y);
      
      Dcb[iz*nk + ik] = y[2*N_tau*N_mu];
      fcb[iz*nk + ik] = y[2*N_tau*N_mu+1]/y[2*N_tau*N_mu];
      Dnu[iz*nk + ik] = y[2*N_mu]; //d_nu_mono(z_n[iz],y);
      Tcb[iz*nk + ik] = y[2*N_tau*N_mu+1];
      dcb[iz*nk + ik] = y[2*N_tau*N_mu];
    }
    
    //normalize growths
    if(z_n[nz-1] > 0) evolve_step(k,z_n[nz-1],0,y);
    for(int iz=0; iz<nz; iz++){
      Dnu[iz*nk + ik] /= y[2*N_tau*N_mu]; //Dcb[(nz-1)*nk + ik];
      Dcb[iz*nk + ik] /= y[2*N_tau*N_mu]; //Dcb[(nz-1)*nk + ik];
    }

    if(OUTPUT_STATUS){
      printf("#compute_growths: ik = %i / %i done with norm %g\n",
	     ik, nk, y[2*N_tau*N_mu]);
      fflush(stdout);
    }
    
  }//end parallel for
  
  return 0;
}

double eta_convert (double a) {
    return log(a/aeta_in);
}

//////////////////////////////////// MAIN //////////////////////////////////////
//various tests of code

int main(int argn, char *args[]){

  //initialize
  tau_t_eV(0);

  //compute nu and cb growth factors and rates over a range of k in parallel
    
  const int nk=128, nz=3; // set the number of k modes and number of redshifts you want
  const double z_n[] = {99, 49, 0}; // manually enter the redshifts you want, size of z_n must be the same as nz. z=0 must be in the list for normalisation purposes.
  const double kmin=1e-4, kmax=20; // set the max and min values of k range you wish to cover
  double dlnk=log(kmax/kmin)/(nk-1), k_n[nk]; // populate the k_n array with nk values log sampled over kmin and kmax
  for(int i=0; i<nk; i++) k_n[i] = kmin*exp(dlnk*i);
    double Dnu[nz*nk], Dcb[nz*nk], fcb[nz*nk], Tcb[nz*nk], dcb[nz*nk];; // declare arrays to hold the relevant info. Dnu = neutrino growth factor; Dcb = cb growth factor; fcb = cb growth rate; Tcb = cb velocity divergence; dcb = cb density contrast (unnormalised growth);

    compute_growths(nz,z_n,nk,k_n,Dcb,fcb,Dnu,Tcb,dcb); // call the growth function with the pointers to the above arrays
    
    for(int iz=0; iz<nz; iz++){ //output results
      for(int ik=0; ik<nk; ik++){
        int izk = iz*nk + ik;
          printf("%g %g %g %g %g %g %g\n",z_n[iz],k_n[ik],Dcb[izk],fcb[izk],Dnu[izk],Tcb[izk],dcb[izk]);
        }
      printf("\n\n");
    } 
   
  tau_t_eV(FREE_TAU_TABLE);
  return 0;
}
